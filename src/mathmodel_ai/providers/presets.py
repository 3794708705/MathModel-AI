from dataclasses import dataclass, field

from mathmodel_ai.schemas.provider_registry import (
    CapabilityEvidence,
    CapabilitySource,
    CapabilityStatus,
    EndpointTrustLevel,
    ModelCapability,
    ProviderEndpoint,
    ProviderProtocol,
)


@dataclass(frozen=True)
class ProviderPreset:
    preset_id: str
    display_name: str
    protocol: ProviderProtocol
    base_url: str | None
    recommended_credential_ref: str
    trust_level: EndpointTrustLevel
    credential_type: str = "API_KEY"
    model_hints: tuple[str, ...] = ()
    capability_hints: dict[ModelCapability, CapabilityEvidence] = field(default_factory=dict)

    def endpoint(
        self,
        *,
        provider_id: str | None = None,
        base_url: str | None = None,
    ) -> ProviderEndpoint:
        resolved_url = base_url or self.base_url
        if resolved_url is None:
            raise ValueError(f"preset {self.preset_id} requires an explicit base_url")
        return ProviderEndpoint(
            provider_id=provider_id or self.preset_id.replace("_", "-"),
            display_name=self.display_name,
            protocol=self.protocol,
            base_url=resolved_url,
            credential_ref=self.recommended_credential_ref,
            trust_level=self.trust_level,
        )


DEEPSEEK_OFFICIAL = ProviderPreset(
    preset_id="deepseek_official",
    display_name="DeepSeek Official API",
    protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
    base_url="https://api.deepseek.com",
    recommended_credential_ref="env:DEEPSEEK_API_KEY",
    trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    model_hints=("deepseek-v4-flash", "deepseek-v4-pro"),
    capability_hints={
        ModelCapability.TEXT: CapabilityEvidence(
            status=CapabilityStatus.PARTIAL,
            source=CapabilitySource.UNKNOWN,
            detail="preset compatibility hint only; run the model capability probe",
        )
    },
)

QWEN_MODELSTUDIO = ProviderPreset(
    preset_id="qwen_modelstudio",
    display_name="Qwen Model Studio",
    protocol=ProviderProtocol.OPENAI_CHAT_COMPLETIONS,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    recommended_credential_ref="env:DASHSCOPE_API_KEY",
    trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    model_hints=("qwen-plus", "qwen-max"),
    capability_hints={
        ModelCapability.TEXT: CapabilityEvidence(
            status=CapabilityStatus.PARTIAL,
            source=CapabilitySource.UNKNOWN,
            detail="preset compatibility hint only; run the model capability probe",
        )
    },
)

OPENAI_OFFICIAL = ProviderPreset(
    preset_id="openai_official",
    display_name="OpenAI",
    protocol=ProviderProtocol.OPENAI_RESPONSES,
    base_url="https://api.openai.com/v1",
    recommended_credential_ref="env:OPENAI_API_KEY",
    trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    model_hints=("gpt-5", "gpt-4.1"),
)

GOOGLE_AI = ProviderPreset(
    preset_id="google_ai",
    display_name="Google AI",
    protocol=ProviderProtocol.GOOGLE_GENERATE_CONTENT,
    base_url="https://generativelanguage.googleapis.com/v1beta",
    recommended_credential_ref="env:GOOGLE_API_KEY",
    trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    model_hints=("gemini-2.5-pro", "gemini-2.5-flash"),
)

ANTHROPIC_OFFICIAL = ProviderPreset(
    preset_id="anthropic_official",
    display_name="Anthropic",
    protocol=ProviderProtocol.ANTHROPIC_MESSAGES,
    base_url="https://api.anthropic.com/v1",
    recommended_credential_ref="env:ANTHROPIC_API_KEY",
    trust_level=EndpointTrustLevel.OFFICIAL_VENDOR,
    model_hints=("claude-sonnet", "claude-opus"),
)

PROVIDER_PRESETS = {
    item.preset_id: item
    for item in (
        DEEPSEEK_OFFICIAL,
        QWEN_MODELSTUDIO,
        OPENAI_OFFICIAL,
        GOOGLE_AI,
        ANTHROPIC_OFFICIAL,
    )
}
