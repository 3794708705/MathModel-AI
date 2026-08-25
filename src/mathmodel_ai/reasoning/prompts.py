from importlib.resources import files
from string import Template

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)

    def render_user(self, **values: str) -> str:
        return Template(self.user).substitute(values)


class PromptRegistry:
    """Loads immutable, versioned prompt resources shipped with the package."""

    def __init__(self, package: str = "mathmodel_ai.prompt_templates") -> None:
        self._package = package
        self._cache: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate:
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        resource = files(self._package).joinpath(f"{name}.prompt")
        if not resource.is_file():
            raise KeyError(f"unknown prompt template: {name}")
        raw = resource.read_text(encoding="utf-8")
        header, separator, body = raw.partition("\n---\n")
        if not separator:
            raise ValueError(f"prompt {name!r} has no metadata separator")
        metadata = dict(line.split(":", maxsplit=1) for line in header.splitlines() if ":" in line)
        system, user_separator, user = body.partition("\n---USER---\n")
        if not user_separator:
            raise ValueError(f"prompt {name!r} has no user section")
        prompt = PromptTemplate(
            name=metadata.get("name", name).strip(),
            version=metadata["version"].strip(),
            system=system.strip(),
            user=user.strip(),
        )
        self._cache[name] = prompt
        return prompt
