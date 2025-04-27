import knime.extension as knext
import knime.api.schema as ks
from knime.extension import ConfigurationContext, ExecutionContext
from ._base import (
    hub_connector_icon,
    knime_category,
    create_authorization_headers,
    extract_api_base,
    create_model_choice_provider,
    list_models,
    validate_auth_spec,
)


from ..base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)


class KnimeHubEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        auth_spec: ks.HubAuthenticationPortObjectSpec,
        model_name: str,
    ) -> None:
        super().__init__()
        self._auth_spec = auth_spec
        self._model_name = model_name

    @property
    def auth_spec(self) -> ks.HubAuthenticationPortObjectSpec:
        return self._auth_spec

    @property
    def model_name(self) -> str:
        return self._model_name

    def validate_context(self, ctx: ConfigurationContext):
        validate_auth_spec(self.auth_spec)

    def serialize(self) -> dict:
        return {
            "auth": self.auth_spec.serialize(),
            "model_name": self.model_name,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            ks.HubAuthenticationPortObjectSpec.deserialize(data["auth"]),
            data["model_name"],
        )


class KnimeHubEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> KnimeHubEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: ExecutionContext):
        from ._embeddings_model import OpenAIEmbeddings

        auth_spec = self.spec.auth_spec
        return OpenAIEmbeddings(
            model=self.spec.model_name,
            base_url=extract_api_base(auth_spec),
            api_key="placeholder",
            extra_headers=create_authorization_headers(auth_spec),
        )


knime_embeddings_port_type = knext.port_type(
    "KNIME Hub Embeddings",
    KnimeHubEmbeddingsPortObject,
    KnimeHubEmbeddingsPortObjectSpec,
)


@knext.node(
    name="KNIME Hub Embeddings Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "GenAI Gateway",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_port(
    name="KNIME Hub Embeddings",
    description="An embeddings model that connects to a KNIME Hub to embed documents.",
    port_type=knime_embeddings_port_type,
)
class KnimeHubEmbeddingsConnector:
    """
    Connects to an embedding model configured in the GenAI Gateway of the connected KNIME Hub.

    Connects to an embeddings model configured in the GenAI Gateway of the connected KNIME Hub using the authentication
    provided via the input port.

    Use this node to generate embeddings, which are dense vector representations of text input data.
    Embeddings are useful for tasks like similarity search, e.g. in a retrieval augmented generation (RAG) system but
    can also be used for clustering, classification and other machine learning applications.
    """

    model_name = knext.StringParameter(
        "Model",
        "Select the model to use.",
        choices=create_model_choice_provider("embedding"),
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)
        return self._create_spec(authentication)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> KnimeHubEmbeddingsPortObject:
        available_models = list_models(authentication.spec, "embedding")
        if self.model_name not in available_models:
            raise knext.InvalidParametersError(
                f"The selected model {self.model_name} is not served by the connected Hub."
            )
        return KnimeHubEmbeddingsPortObject(self._create_spec(authentication.spec))

    def _create_spec(
        self, authentication: ks.HubAuthenticationPortObjectSpec
    ) -> KnimeHubEmbeddingsPortObjectSpec:
        return KnimeHubEmbeddingsPortObjectSpec(
            authentication,
            self.model_name,
        )
