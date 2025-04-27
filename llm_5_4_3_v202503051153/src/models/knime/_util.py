import knime.extension as knext
import knime.api.schema as ks
import util
from ._base import (
    hub_connector_icon,
    knime_category,
    list_models_with_descriptions,
    validate_auth_spec,
)
import pyarrow as pa


@knext.parameter_group("Model types")
class ModelTypeSettings:
    chat_models = knext.BoolParameter(
        "Chat models",
        "List available chat models.",
        True,
    )

    embedding_models = knext.BoolParameter(
        "Embedding models",
        "List available embedding models.",
        True,
    )


@knext.node(
    name="KNIME Hub AI Model Lister",
    node_type=knext.NodeType.SOURCE,
    icon_path=hub_connector_icon,
    category=knime_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway", "LLM", "RAG"],
)
@knext.input_port(
    name="KNIME Hub Credential",
    description="Credential for a KNIME Hub.",
    port_type=knext.PortType.HUB_AUTHENTICATION,
)
@knext.output_table(
    name="KNIME Hub AI Model list",
    description="The list of models, including their name, type and description.",
)
class KnimeHubAIModelLister:
    """
    Lists available models in the GenAI Gateway of the connected KNIME Hub.

    Lists available models in the GenAI Gateway of the connected KNIME Hub using the
    authentication provided via the input port.

    Use this node to retrieve the available models with their name, type and description.
    """

    model_types = ModelTypeSettings()

    # Name, KNIME type, and PyArrow type of the columns to output
    column_list = [
        util.OutputColumn("Name", knext.string(), pa.string()),
        util.OutputColumn("Type", knext.string(), pa.string()),
        util.OutputColumn("Description", knext.string(), pa.string()),
    ]

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication: ks.HubAuthenticationPortObjectSpec,
    ) -> knext.Schema:
        # raises exception if the hub authenticator has not been executed
        validate_auth_spec(authentication)

        knime_columns = [column.to_knime_column() for column in self.column_list]

        return knext.Schema.from_columns(knime_columns)

    def execute(
        self, ctx: knext.ExecutionContext, authentication: knext.PortObject
    ) -> knext.Table:
        import pandas as pd

        available_models = []

        if self.model_types.chat_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "chat")
            )

        if self.model_types.embedding_models:
            available_models.extend(
                list_models_with_descriptions(authentication.spec, "embedding")
            )

        if not available_models:
            return self._create_empty_table()

        models_df = pd.DataFrame(
            available_models, columns=["Name", "Type", "Description"]
        )

        return knext.Table.from_pandas(models_df)

    def _create_empty_table(self) -> knext.Table:
        """Constructs an empty KNIME Table with the correct output columns."""

        return util.create_empty_table(None, self.column_list)
