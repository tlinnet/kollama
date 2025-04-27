import knime.extension as knext
# These import has to be relative. When using "src.models.ollama" the nodes disappear in the KNIME GUI.
from ._util import ollama_icon, ollama_category
from .auth import ollama_auth_port_type
from ._auth import OllamaAuthenticationPortObject, OllamaAuthenticationPortObjectSpec
from ._embeddings import OllamaEmbeddingsPortObject, OllamaEmbeddingsPortObjectSpec


ollama_embeddings_port_type = knext.port_type(
    "Ollama Embeddings", OllamaEmbeddingsPortObject, OllamaEmbeddingsPortObjectSpec,
)

def _list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_model_list(ctx, mode="embedding")
    return ["ollama-chat", "ollama-reasoner"]


@knext.node(
    name="Ollama Embeddings Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=ollama_icon,
    category=ollama_category,
    keywords=["Ollama", "GenAI", "RAG"],
)
@knext.input_port(
    "Ollama Authentication",
    "The authentication for the Ollama API.",
    ollama_auth_port_type,
)
@knext.output_port(
    name="Ollama Embeddings",
    description="An embeddings model that connects to a Ollama to embed documents.",
    port_type=ollama_embeddings_port_type,
)
class OllamaEmbeddingsConnector:
    """Connects to a embedding model provided by the Ollama API.

    Use this node to generate embeddings, which are dense vector representations of text input data.
    Embeddings are useful for tasks like similarity search, e.g. in a retrieval augmented generation (RAG) system but
    can also be used for clustering, classification and other machine learning applications.

    """

    model = knext.StringParameter(
        "Model",
        description="The model to use. The available models are fetched from the Ollama API if possible.",
        default_value="nomic-embed-text:latest",
        choices=_list_models,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: OllamaAuthenticationPortObjectSpec,
    ) -> OllamaEmbeddingsPortObjectSpec:
        auth.validate_context(ctx)
        return self._create_spec(auth)

    def _create_spec(
        self, auth: OllamaAuthenticationPortObjectSpec,
    ) -> OllamaEmbeddingsPortObjectSpec:
        return OllamaEmbeddingsPortObjectSpec(
            auth=auth,
            model=self.model,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: OllamaAuthenticationPortObject
    ) -> OllamaEmbeddingsPortObject:
        return OllamaEmbeddingsPortObject(self._create_spec(auth.spec))

