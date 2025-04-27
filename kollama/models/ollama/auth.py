import knime.extension as knext

from kollama.models.ollama._util import ollama_icon, ollama_category
from kollama.models.ollama._auth import _default_ollama_api_base, _default_ollama_timeout, OllamaAuthenticationPortObject, OllamaAuthenticationPortObjectSpec


ollama_auth_port_type = knext.port_type(
    "Ollama Authentication",
    OllamaAuthenticationPortObject,
    OllamaAuthenticationPortObjectSpec,
)


@knext.node(
    name="Ollama Authenticator",
    node_type=knext.NodeType.SOURCE,
    icon_path=ollama_icon,
    category=ollama_category,
    keywords=["Ollama", "GenAI"],
)
@knext.output_port(
    "Ollama API Authentication",
    "Authentication for the Ollama API",
    ollama_auth_port_type,
)
class OllamaAuthenticator:
    """Authenticates with the Ollama API via API key.

    **Note**: Default installation of Ollama has no API key.
    """

    base_url = knext.StringParameter(
        "Base URL",
        "The base URL of the Ollama API.",
        default_value=_default_ollama_api_base,
        is_advanced=False,
    )

    timeout = knext.IntParameter(
        "Timeout",
        "The timeout of the Ollama API.",
        default_value=_default_ollama_timeout,
        is_advanced=False,
    )

    validate_api_connection = knext.BoolParameter(
        "Validate API Connection",
        "If set, the API connection is validated during execution by fetching the available models.",
        True,
        is_advanced=False,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> OllamaAuthenticationPortObjectSpec:
        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext) -> OllamaAuthenticationPortObject:
        spec = self.create_spec()
        if self.validate_api_connection:
            spec.validate_api_connection(ctx)
        return OllamaAuthenticationPortObject(spec)

    def create_spec(self) -> OllamaAuthenticationPortObjectSpec:
        return OllamaAuthenticationPortObjectSpec(
            base_url=self.base_url, timeout=self.timeout
        )
