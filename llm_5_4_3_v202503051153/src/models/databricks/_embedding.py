from typing import Dict
import knime.extension as knext

from ._utils import (
    check_workspace_available,
    get_api_key,
    get_base_url,
    get_model_choices_provider,
    get_models,
    get_workspace_port_type,
    databricks_category,
)
from ..base import EmbeddingsPortObjectSpec, EmbeddingsPortObject

databricks_workspace_port_type = get_workspace_port_type()


class DatabricksEmbeddingPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self, databricks_workspace_spec: knext.CredentialPortObjectSpec, endpoint: str
    ):
        super().__init__()
        self._databricks_workspace_spec = databricks_workspace_spec
        self._endpoint = endpoint

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def databricks_workspace_spec(self) -> knext.CredentialPortObjectSpec:
        return self._databricks_workspace_spec

    def validate_context(self, ctx):
        check_workspace_available(self._databricks_workspace_spec)

    def serialize(self):
        return {
            "databricks_workspace_spec": self._databricks_workspace_spec.serialize(),
            "endpoint": self._endpoint,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        return cls(
            databricks_workspace_spec=databricks_workspace_port_type.spec_class.deserialize(
                data["databricks_workspace_spec"]
            ),
            endpoint=data["endpoint"],
        )


class DatabricksEmbeddingPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> DatabricksEmbeddingPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from ..knime._embeddings_model import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=get_api_key(self.spec.databricks_workspace_spec),
            base_url=get_base_url(self.spec.databricks_workspace_spec),
            model=self.spec.endpoint,
        )


databricks_embedding_port_type = knext.port_type(
    "Databricks Embedding Model",
    DatabricksEmbeddingPortObject,
    DatabricksEmbeddingPortObjectSpec,
)


@knext.node(
    name="Databricks Embedding Connector",
    category=databricks_category,
    icon_path="icons/Databricks-embeddings-connector.png",
    keywords=["Databricks", "Embedding", "GenAI", "Mosaic"],
    node_type=knext.NodeType.SOURCE,
)
@knext.input_port(
    "Databricks Workspace",
    "Credentials for a Databricks workspace.",
    databricks_workspace_port_type,
)
@knext.output_port(
    "Databricks Embedding Model",
    "Connection to an embedding model served by a Databricks workspace.",
    databricks_embedding_port_type,
)
class DatabricksEmbeddingConnector:
    """Connects to an embedding model served by a Databricks workspace.

    This node connects to an embedding model served by the Databricks workspace that is provided as an input.
    See the
    [Databricks documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html)
    for more information on how to serve a model in a Databricks workspace.

    **Note**: This node is only available if the
    [KNIME Databricks Integration](https://hub.knime.com/knime/extensions/org.knime.features.bigdata.databricks/latest)
    is installed.
    """

    endpoint = knext.StringParameter(
        "Endpoint",
        "The name of the endpoint of the model in the Databricks workspace.",
        default_value="",
        choices=get_model_choices_provider("embeddings"),
    )

    def configure(self, ctx, databricks_workspace_spec):
        if self.endpoint == "":
            raise knext.InvalidParametersError("Select an embedding model endpoint.")
        check_workspace_available(databricks_workspace_spec)
        return self._create_spec(databricks_workspace_spec)

    def _create_spec(
        self, databricks_workspace_spec
    ) -> DatabricksEmbeddingPortObjectSpec:
        return DatabricksEmbeddingPortObjectSpec(
            databricks_workspace_spec=databricks_workspace_spec,
            endpoint=self.endpoint,
        )

    def execute(self, ctx, databricks_workspace) -> DatabricksEmbeddingPortObject:
        if self.endpoint not in get_models(databricks_workspace.spec, "embeddings"):
            raise knext.InvalidParametersError(
                f"The embedding model '{self.endpoint}' is not served by the Databricks workspace."
            )
        return DatabricksEmbeddingPortObject(
            self._create_spec(databricks_workspace.spec)
        )
