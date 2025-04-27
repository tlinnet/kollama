import knime.extension as knext

from kollama.models.ollama._util import ollama_icon, ollama_category
from kollama.models.ollama.auth import ollama_auth_port_type
from kollama.models.ollama._auth import OllamaAuthenticationPortObject, OllamaAuthenticationPortObjectSpec
from kollama.models.ollama._chat import OllamaChatModelPortObject, OllamaChatModelPortObjectSpec


ollama_chat_model_port_type = knext.port_type(
    "Ollama Chat Model", OllamaChatModelPortObject, OllamaChatModelPortObjectSpec
)

def _list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_model_list(ctx, mode="chat")
    return ["ollama-chat", "ollama-reasoner"]


@knext.node(
    name="Ollama Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=ollama_icon,
    category=ollama_category,
    keywords=["Ollama", "GenAI", "Reasoning"],
)
@knext.input_port(
    "Ollama Authentication",
    "The authentication for the Ollama API.",
    ollama_auth_port_type,
)
@knext.output_port(
    "Ollama Chat Model",
    "The Ollama chat model which can be used in the LLM Prompter and Chat Model Prompter.",
    ollama_chat_model_port_type,
)
class OllamaChatModelConnector:
    """Connects to a chat model provided by the Ollama API.

    This node establishes a connection with a Ollama Chat Model. After successfully authenticating
    using the **Ollama Authenticator** node, you can select a chat model from a predefined list.

    **Note**: Default installation of Ollama has no API key.
    """

    model = knext.StringParameter(
        "Model",
        description="The model to use. The available models are fetched from the Ollama API if possible.",
        default_value="ollama-chat",
        choices=_list_models,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0.

        Higher values will lead to less deterministic but more creative answers.
        Recommended values for different tasks:

        - Coding / math: 0.0
        - Data cleaning / data analysis: 1.0
        - General conversation: 1.3
        - Translation: 1.3
        - Creative writing: 1.5
        """,
        default_value=1,
    )

    num_predict = knext.IntParameter(
        "Num Predict",
        description="The maximum number of tokens to generate in the response",
        default_value=4096,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: OllamaAuthenticationPortObjectSpec,
    ) -> OllamaChatModelPortObjectSpec:
        auth.validate_context(ctx)
        return self.create_spec(auth)

    def create_spec(
        self, auth: OllamaAuthenticationPortObjectSpec
    ) -> OllamaChatModelPortObjectSpec:
        return OllamaChatModelPortObjectSpec(
            auth=auth,
            model=self.model,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: OllamaAuthenticationPortObject
    ) -> OllamaChatModelPortObject:
        return OllamaChatModelPortObject(self.create_spec(auth.spec))
