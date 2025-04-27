import knime.extension as knext
import pyarrow as pa
import pandas as pd
import json

from kollama.models.ollama._util import ollama_icon, ollama_category
from kollama.models.ollama.auth import ollama_auth_port_type
from kollama.models.ollama._auth import OllamaAuthenticationPortObject, OllamaAuthenticationPortObjectSpec
from kollama.models.ollama._util import OllamaUtil
from kollama.util import create_empty_table, OutputColumn


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
    name="Ollama List",
    node_type=knext.NodeType.SOURCE,
    icon_path=ollama_icon,
    category=ollama_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "GenAI Gateway", "LLM", "RAG"],
)
@knext.input_port(
    name="Ollama Authentication",
    description="The authentication for the Ollama API.",
    port_type=ollama_auth_port_type,
)
@knext.output_table(
    name="Ollama list of models",
    description="The list of models, including their name, type and description.",
)
class OllamaModelLister:
    """
    Lists available models in Ollama.

    Lists available models in Ollama using the
    authentication provided via the input port.

    Use this node to retrieve the available models with their name, type and description.
    """

    model_types = ModelTypeSettings()

    # Name, KNIME type, and PyArrow type of the columns to output
    column_list = [
        OutputColumn("Name", knext.string(), pa.string()),
        OutputColumn("Type", knext.string(), pa.string()),
        OutputColumn("Description", knext.string(), pa.string()),
    ]

    def configure(self,
        ctx: knext.ConfigurationContext,
        auth: OllamaAuthenticationPortObjectSpec,
    ) -> knext.Schema:
        auth.validate_context(ctx)
        knime_columns = [column.to_knime_column() for column in self.column_list]
        return knext.Schema.from_columns(knime_columns)

    def execute(self, 
        ctx: knext.ExecutionContext, 
        auth: OllamaAuthenticationPortObject
    ) -> knext.Table:
        available_models = {}
        returned_models = []
        util = OllamaUtil(base_url=auth.spec.base_url, timeout=auth.spec.timeout)

        if self.model_types.chat_models:
            available_models.update(util.ollama_list_models(mode="chat", verbose=True))

        if self.model_types.embedding_models:
            available_models.update(util.ollama_list_models(mode="embedding", verbose=True))

        for model in available_models:
            returned_models.append([model, available_models[model]["type"], json.dumps(available_models[model])])

        if not returned_models:
            return self._create_empty_table()

        models_df = pd.DataFrame(
            returned_models, columns=["Name", "Type", "Description"]
        )

        return knext.Table.from_pandas(models_df)

    def _create_empty_table(self) -> knext.Table:
        """Constructs an empty KNIME Table with the correct output columns."""

        return create_empty_table(None, self.column_list)
