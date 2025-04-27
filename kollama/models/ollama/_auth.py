import knime.extension as knext

from kollama.base import AIPortObjectSpec
from kollama.models.ollama._util import OllamaUtil

_default_ollama_api_base = "http://localhost:11434"
_default_ollama_timeout = 5


class OllamaAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(self,
        base_url: str = _default_ollama_api_base,
        timeout: int = _default_ollama_timeout
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    def validate_context(self, ctx: knext.ConfigurationContext):
        if not self.base_url:
            raise knext.InvalidParametersError("Please provide a base URL.")

    def validate_api_connection(self, ctx: knext.ExecutionContext):
        try:
            self._get_models_from_api(ctx)
        except Exception as e:
            raise RuntimeError(f"Could not access Ollama API at {self.base_url}") from e

    def _get_models_from_api(
        self, ctx: knext.ConfigurationContext | knext.ExecutionContext, 
        mode: str | None = None
    ) -> list[str]:
        util = OllamaUtil(base_url=self.base_url, timeout=self.timeout)
        models = util.ollama_list_models(mode=mode)
        models_names = sorted(models.keys())
        return models_names

    def get_model_list(self, ctx: knext.ConfigurationContext, mode: str | None = None) -> list[str]:
        try:
            return self._get_models_from_api(ctx, mode=mode)
        except Exception:
            return ["ollama-chat", "ollama-embedding"]

    def serialize(self) -> dict:
        return {
            "base_url": self.base_url,
            "timeout": self.timeout
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            base_url=data.get("base_url", _default_ollama_api_base),
            timeout=data.get("timeout", _default_ollama_timeout)
        )


class OllamaAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: OllamaAuthenticationPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> OllamaAuthenticationPortObjectSpec:
        return super().spec

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: OllamaAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)
