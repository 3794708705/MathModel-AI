from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mathmodel_ai.core.types import Environment, ProviderName, ReasoningEffort


class ModelTargetSettings(BaseModel):
    provider: ProviderName
    model: str = Field(min_length=1)
    reasoning: ReasoningEffort


class ModelCatalogSettings(BaseModel):
    fast: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.OPENAI,
            model="gpt-5.6-luna",
            reasoning=ReasoningEffort.LOW,
        )
    )
    balanced: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.OPENAI,
            model="gpt-5.6-terra",
            reasoning=ReasoningEffort.MEDIUM,
        )
    )
    flagship_high: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.OPENAI,
            model="gpt-5.6-sol",
            reasoning=ReasoningEffort.HIGH,
        )
    )
    flagship_xhigh: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.OPENAI,
            model="gpt-5.6-sol",
            reasoning=ReasoningEffort.XHIGH,
        )
    )
    flagship_max: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.OPENAI,
            model="gpt-5.6-sol",
            # The public Responses API currently documents xhigh, not max.
            reasoning=ReasoningEffort.XHIGH,
        )
    )
    multimodal: ModelTargetSettings = Field(
        default_factory=lambda: ModelTargetSettings(
            provider=ProviderName.GOOGLE,
            model="gemini-3.7-flash",
            reasoning=ReasoningEffort.XHIGH,
        )
    )


class ModelJuryWeightSettings(BaseModel):
    problem_fit: int = Field(default=25, ge=0)
    data_fit: int = Field(default=15, ge=0)
    mathematical_validity: int = Field(default=15, ge=0)
    explainability: int = Field(default=10, ge=0)
    validation_potential: int = Field(default=10, ge=0)
    innovation_potential: int = Field(default=10, ge=0)
    competition_feasibility: int = Field(default=10, ge=0)
    computational_cost: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def weights_sum_to_one_hundred(self) -> "ModelJuryWeightSettings":
        if sum(self.model_dump().values()) != 100:
            raise ValueError("model jury weights must sum to 100")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "MathModel AI"
    app_version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel"
    database_echo: bool = False
    default_provider: ProviderName = ProviderName.MOCK
    default_provider_model: str = Field(default="mock-foundation", min_length=1)
    provider_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    reasoning_max_retries: int = Field(default=2, ge=0, le=5)
    ambiguity_review_threshold: float = Field(default=0.7, ge=0, le=1)
    storage_root: Path = Path("var/storage")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_archive_entries: int = Field(default=10_000, ge=10)
    max_archive_uncompressed_bytes: int = Field(default=250 * 1024 * 1024, ge=1024)
    max_archive_compression_ratio: float = Field(default=100.0, ge=1)
    max_image_pixels: int = Field(default=50_000_000, ge=1_000_000)
    max_pdf_pages: int = Field(default=500, ge=1, le=10_000)
    max_pdf_page_images: int = Field(default=10, ge=0, le=100)
    max_multimodal_inline_bytes: int = Field(default=18 * 1024 * 1024, ge=1024, le=19 * 1024 * 1024)

    sandbox_image: str = "mathmodel-ai-sandbox:phase3"
    sandbox_root: Path = Path("var/sandbox")
    sandbox_cpu_cores: float = Field(default=1.0, gt=0, le=8)
    sandbox_memory_mb: int = Field(default=512, ge=64, le=16_384)
    sandbox_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    sandbox_pids_limit: int = Field(default=64, ge=16, le=1024)
    sandbox_max_output_bytes: int = Field(default=1_000_000, ge=1024)
    sandbox_max_artifacts: int = Field(default=50, ge=0, le=1000)
    sandbox_max_artifact_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    solver_sandbox_image: str = "mathmodel-ai-solver:phase4"
    solver_sandbox_root: Path = Path("var/solver-sandbox")
    gurobi_sandbox_image: str | None = None
    gurobi_license_file: Path | None = None
    lp_solver_preference: list[str] = Field(default_factory=lambda: ["SCIPY_HIGHS", "GUROBI"])
    milp_solver_preference: list[str] = Field(
        default_factory=lambda: ["GUROBI", "SCIPY_MILP", "ORTOOLS_CP_SAT"]
    )
    integer_solver_preference: list[str] = Field(
        default_factory=lambda: ["ORTOOLS_CP_SAT", "GUROBI", "SCIPY_MILP"]
    )
    nlp_solver_preference: list[str] = Field(default_factory=lambda: ["SCIPY_MINIMIZE"])
    evidence_abs_tolerance: float = Field(default=1e-9, ge=0, le=0.1)
    evidence_rel_tolerance: float = Field(default=1e-9, ge=0, le=0.1)
    solver_tiny_max_variables: int = Field(default=10, ge=1)
    solver_tiny_max_constraints: int = Field(default=10, ge=1)
    solver_tiny_max_nonzeros: int = Field(default=100, ge=1)
    solver_small_max_variables: int = Field(default=100, ge=1)
    solver_small_max_constraints: int = Field(default=100, ge=1)
    solver_small_max_nonzeros: int = Field(default=2_000, ge=1)
    solver_medium_max_variables: int = Field(default=1_000, ge=1)
    solver_medium_max_constraints: int = Field(default=1_000, ge=1)
    solver_medium_max_nonzeros: int = Field(default=100_000, ge=1)
    solver_deadline_pressure_3_seconds: float = Field(default=300, gt=0, le=3600)
    solver_deadline_pressure_4_seconds: float = Field(default=120, gt=0, le=3600)
    solver_deadline_pressure_5_seconds: float = Field(default=60, gt=0, le=3600)

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    google_api_key: SecretStr | None = None
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"

    model_catalog: ModelCatalogSettings = Field(default_factory=ModelCatalogSettings)
    model_jury_weights: ModelJuryWeightSettings = Field(default_factory=ModelJuryWeightSettings)

    @model_validator(mode="after")
    def disallow_mock_in_production(self) -> "Settings":
        if (
            self.environment is Environment.PRODUCTION
            and self.default_provider is ProviderName.MOCK
        ):
            raise ValueError("MM_DEFAULT_PROVIDER=mock is forbidden in production")
        required_key = {
            ProviderName.OPENAI: self.openai_api_key,
            ProviderName.GOOGLE: self.google_api_key,
            ProviderName.ANTHROPIC: self.anthropic_api_key,
            ProviderName.MOCK: None,
        }[self.default_provider]
        if self.environment is Environment.PRODUCTION and required_key is None:
            raise ValueError(
                f"an API key is required for production provider {self.default_provider.value}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
