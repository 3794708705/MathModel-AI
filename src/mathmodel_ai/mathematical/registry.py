from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from mathmodel_ai.mathematical.expressions import referenced_symbols
from mathmodel_ai.schemas.mathematical import (
    EquationDefinition,
    MathematicalModel,
    MathExpression,
    ParameterDefinition,
    UnitExpression,
    VariableDefinition,
    VariableDomain,
)


class RegistryIssueCode(StrEnum):
    UNDEFINED_SYMBOL = "UNDEFINED_SYMBOL"
    DUPLICATE_SYMBOL = "DUPLICATE_SYMBOL"
    CONFLICTING_DEFINITION = "CONFLICTING_DEFINITION"
    UNUSED_CORE_SYMBOL = "UNUSED_CORE_SYMBOL"
    DUPLICATE_EQUATION = "DUPLICATE_EQUATION"
    UNDEFINED_EQUATION_DEPENDENCY = "UNDEFINED_EQUATION_DEPENDENCY"
    DUPLICATE_PARAMETER = "DUPLICATE_PARAMETER"


class RegistryIssue(BaseModel):
    code: RegistryIssueCode
    reference: str
    message: str
    critical: bool = True


class RegistryReport(BaseModel):
    issues: list[RegistryIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(item.critical for item in self.issues)


class SymbolKind(StrEnum):
    VARIABLE = "VARIABLE"
    PARAMETER = "PARAMETER"
    CONSTANT = "CONSTANT"
    SET = "SET"
    INDEX = "INDEX"


class SymbolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    symbol: str
    canonical_name: str
    meaning: str
    unit: UnitExpression | None = None
    domain: VariableDomain | None = None
    kind: SymbolKind
    owner_model: UUID
    owner_version: int = Field(ge=1)
    first_definition: str
    references: list[str] = Field(default_factory=list)
    indexed: bool = False


def _base_symbol(symbol: str) -> str:
    return symbol.partition("[")[0]


class SymbolRegistry:
    def __init__(self, model_id: UUID, version: int) -> None:
        self.model_id = model_id
        self.version = version
        self._symbols: dict[str, SymbolDefinition] = {}
        self._issues: list[RegistryIssue] = []

    @classmethod
    def from_model(cls, model: MathematicalModel) -> SymbolRegistry:
        registry = cls(model.model_id, model.version)
        for variable in [
            *model.decision_variables,
            *model.state_variables,
            *model.derived_variables,
        ]:
            registry.register(registry._from_variable(variable))
        for parameter in model.parameters:
            registry.register(registry._from_parameter(parameter, SymbolKind.PARAMETER))
        for constant in model.constants:
            registry.register(registry._from_parameter(constant, SymbolKind.CONSTANT))
        for model_set in model.sets:
            registry.register(
                SymbolDefinition(
                    symbol=model_set.symbol,
                    canonical_name=model_set.set_id,
                    meaning=model_set.description,
                    kind=SymbolKind.SET,
                    owner_model=model.model_id,
                    owner_version=model.version,
                    first_definition=model_set.set_id,
                )
            )
        for index in model.indices:
            registry.register(
                SymbolDefinition(
                    symbol=index.symbol,
                    canonical_name=index.index_id,
                    meaning=index.description,
                    kind=SymbolKind.INDEX,
                    owner_model=model.model_id,
                    owner_version=model.version,
                    first_definition=index.index_id,
                )
            )
        registry.validate_references(model)
        return registry

    def _from_variable(self, item: VariableDefinition) -> SymbolDefinition:
        return SymbolDefinition(
            symbol=item.symbol,
            canonical_name=item.variable_id,
            meaning=item.description,
            unit=item.unit,
            domain=item.domain,
            kind=SymbolKind.VARIABLE,
            owner_model=self.model_id,
            owner_version=self.version,
            first_definition=item.variable_id,
            indexed=bool(item.index_sets),
        )

    def _from_parameter(self, item: ParameterDefinition, kind: SymbolKind) -> SymbolDefinition:
        return SymbolDefinition(
            symbol=item.symbol,
            canonical_name=item.parameter_id,
            meaning=item.description,
            unit=item.unit,
            kind=kind,
            owner_model=self.model_id,
            owner_version=self.version,
            first_definition=item.parameter_id,
        )

    def register(self, definition: SymbolDefinition) -> None:
        existing = self._symbols.get(definition.symbol)
        if existing is None:
            self._symbols[definition.symbol] = definition
            return
        conflicting = any(
            (
                existing.kind is not definition.kind,
                existing.meaning != definition.meaning,
                existing.unit != definition.unit,
                existing.domain != definition.domain,
            )
        )
        self._issues.append(
            RegistryIssue(
                code=(
                    RegistryIssueCode.CONFLICTING_DEFINITION
                    if conflicting
                    else RegistryIssueCode.DUPLICATE_SYMBOL
                ),
                reference=definition.symbol,
                message=(
                    f"symbol {definition.symbol!r} conflicts between "
                    f"{existing.first_definition} and {definition.first_definition}"
                    if conflicting
                    else f"symbol {definition.symbol!r} is defined more than once"
                ),
            )
        )

    def lookup(self, symbol: str) -> SymbolDefinition | None:
        direct = self._symbols.get(symbol)
        if direct is not None:
            return direct
        base = self._symbols.get(_base_symbol(symbol))
        if base is not None and base.indexed:
            return base
        return None

    def validate_references(self, model: MathematicalModel) -> None:
        expressions: list[tuple[str, MathExpression]] = []
        if model.objective is not None:
            expressions.append((model.objective.objective_id, model.objective.expression))
        for constraint in [
            *model.constraints,
            *model.initial_conditions,
            *model.boundary_conditions,
        ]:
            expressions.extend(
                (
                    (constraint.constraint_id, constraint.expression),
                    (constraint.constraint_id, constraint.rhs),
                )
            )
        for equation in model.equations:
            expressions.extend(
                (
                    (equation.equation_id, equation.lhs),
                    (equation.equation_id, equation.rhs),
                )
            )

        used: set[str] = set()
        undefined: set[str] = set()
        for reference, expression in expressions:
            for symbol in referenced_symbols(expression):
                definition = self.lookup(symbol)
                if definition is None:
                    if symbol not in undefined:
                        self._issues.append(
                            RegistryIssue(
                                code=RegistryIssueCode.UNDEFINED_SYMBOL,
                                reference=symbol,
                                message=f"symbol {symbol!r} is referenced but not defined",
                            )
                        )
                        undefined.add(symbol)
                    continue
                base = definition.symbol
                used.add(base)
                if reference not in definition.references:
                    definition.references.append(reference)

        for symbol, definition in self._symbols.items():
            if (
                definition.kind
                in {
                    SymbolKind.VARIABLE,
                    SymbolKind.PARAMETER,
                    SymbolKind.CONSTANT,
                }
                and symbol not in used
            ):
                self._issues.append(
                    RegistryIssue(
                        code=RegistryIssueCode.UNUSED_CORE_SYMBOL,
                        reference=symbol,
                        message=f"core symbol {symbol!r} is defined but unused",
                        critical=False,
                    )
                )

    @property
    def report(self) -> RegistryReport:
        return RegistryReport(issues=list(self._issues))


class ParameterRegistry:
    def __init__(self, model_id: UUID, version: int) -> None:
        self.model_id = model_id
        self.version = version
        self._parameters: dict[str, ParameterDefinition] = {}
        self._issues: list[RegistryIssue] = []

    @classmethod
    def from_model(cls, model: MathematicalModel) -> ParameterRegistry:
        registry = cls(model.model_id, model.version)
        for parameter in [*model.parameters, *model.constants]:
            registry.register(parameter)
        return registry

    def register(self, parameter: ParameterDefinition) -> None:
        if parameter.symbol in self._parameters:
            self._issues.append(
                RegistryIssue(
                    code=RegistryIssueCode.DUPLICATE_PARAMETER,
                    reference=parameter.symbol,
                    message=f"parameter symbol {parameter.symbol!r} is duplicated",
                )
            )
            return
        self._parameters[parameter.symbol] = parameter

    def lookup(self, symbol: str) -> ParameterDefinition | None:
        return self._parameters.get(symbol)

    @property
    def report(self) -> RegistryReport:
        return RegistryReport(issues=list(self._issues))


class EquationRegistry:
    def __init__(self, model_id: UUID, version: int) -> None:
        self.model_id = model_id
        self.version = version
        self._equations: dict[str, EquationDefinition] = {}
        self._issues: list[RegistryIssue] = []

    @classmethod
    def from_model(cls, model: MathematicalModel) -> EquationRegistry:
        registry = cls(model.model_id, model.version)
        for equation in model.equations:
            registry.register(equation)
        known = set(registry._equations)
        for equation in registry._equations.values():
            for dependency in equation.dependency_refs:
                if dependency not in known:
                    registry._issues.append(
                        RegistryIssue(
                            code=RegistryIssueCode.UNDEFINED_EQUATION_DEPENDENCY,
                            reference=dependency,
                            message=(
                                f"equation {equation.equation_id} depends on unknown "
                                f"equation {dependency}"
                            ),
                        )
                    )
        return registry

    def register(self, equation: EquationDefinition) -> None:
        if equation.equation_id in self._equations:
            self._issues.append(
                RegistryIssue(
                    code=RegistryIssueCode.DUPLICATE_EQUATION,
                    reference=equation.equation_id,
                    message=f"equation {equation.equation_id} is duplicated",
                )
            )
            return
        self._equations[equation.equation_id] = equation

    def lookup(self, equation_id: str) -> EquationDefinition | None:
        return self._equations.get(equation_id)

    @property
    def report(self) -> RegistryReport:
        return RegistryReport(issues=list(self._issues))
