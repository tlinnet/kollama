# KNIME / own imports
from abc import ABC, abstractmethod
import knime.extension as knext
import util
from knime.extension.nodes import (
    FilestorePortObject,
    load_port_object,
    save_port_object,
    PortType,
    get_port_type_for_spec_type,
    get_port_type_for_id,
)
from knime.extension.parameter import ManualFilterConfig
from models.base import (
    EmbeddingsPortObject,
    EmbeddingsPortObjectSpec,
)
from base import AIPortObjectSpec

import pyarrow as pa
from typing import Optional, Any
import os
import shutil
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

store_icon = "icons/store.png"
store_category = knext.category(
    path=util.main_category,
    level_id="stores",
    name="Vector Stores",
    description="",
    icon=store_icon,
)


def handle_missing_metadata_values(
    df,
    metadatas: Optional[list[str]] = None,
):
    # Fills missing metadata values with empty string
    for metadata in metadatas:
        df[metadata] = df[metadata].fillna("")

    return df


class VectorstorePortObjectSpec(AIPortObjectSpec):
    """Marker interface for vector store specs. Used to define the most generic vector store PortType."""

    def __init__(self, metadata_column_names: Optional[list[str]] = None) -> None:
        super().__init__()
        self._metadata_column_names = (
            metadata_column_names if metadata_column_names is not None else []
        )

    @property
    def metadata_column_names(self) -> list[str]:
        return self._metadata_column_names

    def serialize(self) -> dict:
        return {"metadata_column_names": self.metadata_column_names}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data.get("metadata_column_names"))


class VectorstorePortObject(knext.PortObject):
    def __init__(
        self, spec: VectorstorePortObjectSpec, embeddings_model: EmbeddingsPortObject
    ) -> None:
        super().__init__(spec)
        self._embeddings_model = embeddings_model

    @property
    def spec(self) -> VectorstorePortObjectSpec:
        return super().spec

    @property
    def embeddings_model(self) -> EmbeddingsPortObject:
        return self._embeddings_model

    @abstractmethod
    def load_store(self, ctx):
        raise NotImplementedError()

    @abstractmethod
    def get_documents(self, ctx: knext.ExecutionContext) -> tuple[list, np.ndarray]:
        """Retrieves the documents and embeddings"""


vector_store_port_type = knext.port_type(
    "Vectorstore", VectorstorePortObject, VectorstorePortObjectSpec
)


class FilestoreVectorstorePortObjectSpec(VectorstorePortObjectSpec):
    def __init__(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(metadata_column_names)
        self._embeddings_port_type = get_port_type_for_spec_type(type(embeddings_spec))
        self._embeddings_spec = embeddings_spec

    @property
    def embeddings_port_type(self) -> PortType:
        return self._embeddings_port_type

    @property
    def embeddings_spec(self) -> EmbeddingsPortObjectSpec:
        return self._embeddings_spec

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._embeddings_spec.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "embeddings_port_type": self.embeddings_port_type.id,
            "embeddings_spec": self.embeddings_spec.serialize(),
            **super().serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict):
        embeddings_port_type: PortType = get_port_type_for_id(
            data["embeddings_port_type"]
        )
        embeddings_spec = embeddings_port_type.spec_class.deserialize(
            data["embeddings_spec"]
        )
        return cls(embeddings_spec, data.get("metadata_column_names"))


class FilestoreVectorstorePortObject(FilestorePortObject, VectorstorePortObject):
    def __init__(
        self,
        spec: FilestoreVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[Any] = None,
    ):
        super().__init__(spec, embeddings_port_object)
        self._folder_path = folder_path
        self._vectorstore = vectorstore

    def load_store(self, ctx):
        if self._vectorstore is None:
            embeddings = self.embeddings_model.create_model(ctx)
            self._vectorstore = self.load_vectorstore(
                embeddings, self._vectorstore_path, ctx
            )
        return self._vectorstore

    def load_vectorstore(self, embeddings, vectorstore_path, ctx):
        raise NotImplementedError()

    @classmethod
    def _embeddings_path(cls, folder_path: str) -> str:
        return os.path.join(folder_path, "embeddings")

    @property
    def _vectorstore_path(self) -> str:
        return os.path.join(self._folder_path, "vectorstore")

    def write_to(self, file_path):
        os.makedirs(file_path)
        save_port_object(self.embeddings_model, self._embeddings_path(file_path))
        if self._folder_path is not None and not self._folder_path == file_path:
            # copy the folder structures if we have a folder path from read_from
            shutil.copytree(
                self._vectorstore_path, os.path.join(file_path, "vectorstore")
            )
        else:
            self.save_vectorstore(
                os.path.join(file_path, "vectorstore"), self._vectorstore
            )

    def save_vectorstore(self, vectorstore_folder, vectorstore):
        raise NotImplementedError()

    @classmethod
    def _read_embeddings(
        cls, spec: FilestoreVectorstorePortObjectSpec, file_path: str
    ) -> EmbeddingsPortObject:
        return load_port_object(
            spec.embeddings_port_type.object_class,
            spec.embeddings_spec,
            cls._embeddings_path(file_path),
        )

    @classmethod
    def read_from(cls, spec: FilestoreVectorstorePortObjectSpec, file_path: str):
        embeddings_obj = cls._read_embeddings(spec, file_path)
        return cls(spec, embeddings_obj, file_path)

    @abstractmethod
    def get_metadata_filter(self, filter_parameter):
        raise NotImplementedError()


def validate_creator_document_column(input_table: knext.Schema, column: str):
    util.check_column(input_table, column, knext.string(), "document")


@knext.parameter_group(label="Metadata")
class MetadataSettings:
    metadata_columns = knext.ColumnFilterParameter(
        "Metadata columns",
        """Selection of columns used as metadata for each document. The documents column will be ignored.""",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),  # TODO fix typo
        default_value=lambda v: (
            knext.ColumnFilterConfig(
                manual_filter=ManualFilterConfig(include_unknown_columns=False)
            )
            if v < knext.Version(5, 2, 0)
            else knext.ColumnFilterConfig()
        ),
    )


def get_metadata_columns(
    metadata_columns, document_column, schema: knext.Schema
) -> list[str]:
    # metadata was introduced in 5.1.1 and the parameter is None for older versions
    if not metadata_columns:
        return []
    meta_data_columns = [column.name for column in metadata_columns.apply(schema)]
    try:
        meta_data_columns.remove(document_column)
    except:
        pass
    return meta_data_columns


def get_metadata_column_names(ctx: knext.DialogCreationContext) -> list[str]:
    vectorstore_object_spec = ctx.get_input_specs()[0]
    if not vectorstore_object_spec:
        return []
    metadata_column_names = vectorstore_object_spec.metadata_column_names

    return metadata_column_names


@knext.parameter_group(label="Metadata Filter")
class Parameters:
    metadata_column = knext.StringParameter(
        "Metadata column",
        "Metadata column to filter by.",
        choices=get_metadata_column_names,
    )

    metadata_value = knext.StringParameter(
        "Metadata value",
        "Only documents where the specified metadata column matches the given value will be retrieved.",
    )


class BaseVectorStoreCreator(ABC):
    document_column = knext.ColumnParameter(
        "Document column",
        """Select the column containing the documents to be embedded.""",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),
    )

    embeddings_column = knext.ColumnParameter(
        "Embeddings column",
        "Select the column containing existing embeddings if available.",
        port_index=1,
        column_filter=util.create_type_filer(knext.list_(knext.double())),
        include_none_column=True,
        since_version="5.3.2",
        default_value=knext.ColumnParameter.NONE,
    )

    missing_value_handling = knext.EnumParameter(
        "If there are missing values in the document column",
        """Define whether missing values in the document column should be skipped or whether the 
        node execution should fail on missing values.""",
        default_value=lambda v: (
            util.MissingValueHandlingOptions.Fail.name
            if v < knext.Version(5, 2, 0)
            else util.MissingValueHandlingOptions.SkipRow.name
        ),
        enum=util.MissingValueHandlingOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
        since_version="5.2.0",
    )

    metadata_settings = MetadataSettings(since_version="5.2.0")

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
        input_table: knext.Schema,
    ) -> VectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)

        if self.document_column:
            validate_creator_document_column(input_table, self.document_column)
        else:
            self.document_column = util.pick_default_column(input_table, knext.string())

        if self.embeddings_column != knext.ColumnParameter.NONE:
            util.check_column(
                input_table,
                self.embeddings_column,
                knext.list_(knext.double()),
                "embeddings",
            )

        metadata_cols = get_metadata_columns(
            self.metadata_settings.metadata_columns, self.document_column, input_table
        )
        return self._configure(embeddings_spec, metadata_cols)

    @abstractmethod
    def _configure(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: list[str],
    ) -> VectorstorePortObjectSpec:
        """Creates the output spec"""

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> VectorstorePortObject:
        metadata_columns = get_metadata_columns(
            self.metadata_settings.metadata_columns,
            self.document_column,
            input_table.schema,
        )

        df = self._get_relevant_df(input_table, metadata_columns)

        # Skip rows with missing values if "SkipRow" option is selected
        # or fail execution if "Fail" is selected and there are missing documents
        missing_value_handling_setting = util.MissingValueHandlingOptions[
            self.missing_value_handling
        ]

        df = util.handle_missing_and_empty_values(
            df, self.document_column, missing_value_handling_setting, ctx
        )

        embeddings_series = None
        if (
            self.embeddings_column
            and self.embeddings_column != knext.ColumnParameter.NONE
        ):
            df = util.handle_missing_and_empty_values(
                df,
                self.embeddings_column,
                missing_value_handling_setting,
                ctx,
                check_empty_values=False,
            )
            embeddings_series = df[self.embeddings_column]

        def to_document(row):
            from langchain_core.documents import Document

            metadata = {name: row[name] for name in metadata_columns}
            return Document(page_content=row[self.document_column], metadata=metadata)

        documents = df.apply(to_document, axis=1).tolist()

        return self._create_port_object(
            ctx,
            embeddings,
            documents=documents,
            metadata_column_names=metadata_columns,
            embeddings=embeddings_series,
        )

    def _get_relevant_df(
        self,
        input_table: knext.Table,
        metadata_columns: list[str],
    ):
        relevant_columns = [self.document_column] + metadata_columns
        if self.embeddings_column != knext.ColumnParameter.NONE:
            relevant_columns.append(self.embeddings_column)

        return input_table[relevant_columns].to_pandas()

    @abstractmethod
    def _create_port_object(
        self,
        ctx: knext.ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        documents: list,
        metadata_column_names: list[str],
        embeddings,
    ) -> VectorstorePortObject:
        """Creates the vectorstore port object"""


@knext.node(
    "Vector Store Retriever",
    knext.NodeType.SOURCE,
    store_icon,
    category=store_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Vector Store",
    "Vector store containing document embeddings.",
    vector_store_port_type,
)
@knext.input_table(
    "Queries", "Table containing a string column with the queries for the vector store."
)
@knext.output_table(
    "Result table",
    "Table containing the queries and their closest match from the vector store.",
)
class VectorStoreRetriever:
    """
    Performs a similarity search on a vector store.

    This node specializes in retrieving embeddings from a vector store based on their relevance to user queries.

    **Note**: *Dissimilarity scores* calculated using FAISS or Chroma with L2 distance are not bound to a
    specific range, therefore allowing only for ordinal comparison of scores. These scores also depend
    on the embeddings model used to generate the embeddings, as different models produce embeddings
    with varying scales and distributions. Therefore, understanding or comparing similarity across
    different models or spaces without contextual normalization is not meaningful.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    query_column = knext.ColumnParameter(
        "Queries column",
        "Column containing the queries.",
        port_index=1,
        column_filter=util.create_type_filer(knext.string()),
    )

    top_k = knext.IntParameter(
        "Number of results",
        "Number of top results to get from vector store search. Ranking from best to worst.",
        default_value=3,
    )

    retrieved_docs_column_name = knext.StringParameter(
        "Retrieved document column name",
        "The name for the appended column containing the retrieved documents.",
        "Retrieved documents",
    )

    retrieve_metadata = knext.BoolParameter(
        "Retrieve metadata from documents",
        "Whether or not to retrieve document metadata, if provided.",
        default_value=False,
        since_version="5.2.0",
    )

    metadata_filter = knext.ParameterArray(
        label="Metadata Filter",
        description="Select metadata column and value to filter by.",
        since_version="5.4.1",
        parameters=Parameters(),
        button_text="Add filter",
        array_title="Metadata Filter",
    ).rule(knext.OneOf(retrieve_metadata, [True]), knext.Effect.SHOW)

    retrieve_dissimilarity_scores = knext.BoolParameter(
        "Retrieve dissimilarity scores",
        """Whether or not to retrieve dissimilarity scores for the retrieved documents. 
        FAISS and Chroma use L2 distance by default to calculate dissimilarity scores. 
        Lower score represents more similarity.""",
        default_value=False,
        since_version="5.3.0",
    )

    dissimilarity_scores_column_name = knext.StringParameter(
        "Dissimilarity scores column name",
        "The name for the appended column containing the dissimilarity scores.",
        "Dissimilarity scores",
        since_version="5.3.0",
    ).rule(knext.OneOf(retrieve_dissimilarity_scores, [True]), knext.Effect.SHOW)

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vectorstore_spec: VectorstorePortObjectSpec,
        table_spec: knext.Schema,
    ):
        vectorstore_spec.validate_context(ctx)

        if self.query_column:
            util.check_column(table_spec, self.query_column, knext.string(), "queries")
        else:
            self.query_column = util.pick_default_column(table_spec, knext.string())

        if not self.retrieved_docs_column_name:
            raise knext.InvalidParametersError(
                "No name for the column holding the retrieved documents is provided."
            )

        if not self.dissimilarity_scores_column_name:
            raise knext.InvalidParametersError(
                "No name for the column holding the dissimilarity scores is provided."
            )

        output_column_name = self.retrieved_docs_column_name
        table_spec = table_spec.append(
            self._create_column_list(
                table_spec.column_names, vectorstore_spec, output_column_name
            )
        )

        for group in self.metadata_filter:
            if not group.metadata_column or not group.metadata_value:
                raise knext.InvalidParametersError(
                    "Both metadata column and metadata value must be provided for metadata filter."
                )
        return table_spec

    def execute(
        self,
        ctx: knext.ExecutionContext,
        vectorstore: VectorstorePortObject,
        input_table: knext.Table,
    ):
        db = vectorstore.load_store(ctx)
        num_rows = input_table.num_rows
        i = 0
        output_table: knext.BatchOutputTable = knext.BatchOutputTable.create()
        if input_table.num_rows == 0:
            return self._create_empty_table(input_table, vectorstore)

        for batch in input_table.batches():
            doc_collection = []
            dissimilarity_scores = []
            metadata_dict = {}

            df = batch.to_pandas()

            for query in df[self.query_column]:
                util.check_canceled(ctx)

                filter_dict = vectorstore.get_metadata_filter(self.metadata_filter)
                if filter_dict:
                    documents = db.similarity_search_with_score(
                        query, k=self.top_k, filter=filter_dict
                    )
                else:  # Chroma does not support filtering with empty dicts
                    documents = db.similarity_search_with_score(query, k=self.top_k)

                doc_collection.append(
                    [document[0].page_content for document in documents]
                )

                if self.retrieve_metadata:

                    def to_str_or_none(metadata):
                        return str(metadata) if metadata is not None else None

                    for key in vectorstore.spec.metadata_column_names:
                        if key not in metadata_dict:
                            metadata_dict[key] = []
                        metadata_dict[key].append(
                            [
                                to_str_or_none(document[0].metadata.get(key))
                                for document in documents
                            ]
                        )
                if self.retrieve_dissimilarity_scores:
                    dissimilarity_scores.append([document[1] for document in documents])

                i += 1
                ctx.set_progress(i / num_rows)

            output_column_name = util.handle_column_name_collision(
                list(df.columns), self.retrieved_docs_column_name
            )

            df[output_column_name] = doc_collection

            for key in metadata_dict.keys():
                metadata_column_name = util.handle_column_name_collision(
                    list(df.columns), key
                )
                df[metadata_column_name] = metadata_dict[key]

            if self.retrieve_dissimilarity_scores:
                dissimilarity_score_name = util.handle_column_name_collision(
                    list(df.columns), self.dissimilarity_scores_column_name
                )
                df[dissimilarity_score_name] = dissimilarity_scores

            # Check if any documents were retrieved by looking at the length of the retrieved documents
            if df[output_column_name].apply(len).eq(0).all():
                # If no documents were retrieved, each row of all but the first column should be a list containing an empty string
                for column in df.columns[1:]:
                    df[column] = [[""] for _ in range(len(df))]

            output_table.append(df)

        return output_table

    def _create_column_list(
        self, table_spec_columns, vectorstore_spec, output_column_name
    ) -> list[knext.Column]:
        """
        Generates a list of column objects and handles column name collisions for metadata,
        output and dissimilarity score columns.
        Column name collisions are handled in this order:
            - Keep original metadata column names
            - Handle if output_column_name collides with table_spec or metadata column names
            - Handle if dissimilarity_scores_column_name collides with metadata & output column names
        """
        column_names = table_spec_columns
        result_columns = []

        output_column_name = util.handle_column_name_collision(
            column_names, output_column_name
        )

        result_columns.append(
            knext.Column(
                knext.ListType(knext.string()),
                output_column_name,
            )
        )
        column_names.append(output_column_name)

        if self.retrieve_metadata:
            for column_name in vectorstore_spec.metadata_column_names:
                column_name = util.handle_column_name_collision(
                    column_names, column_name
                )
                result_columns.append(
                    knext.Column(knext.ListType(knext.string()), column_name)
                )
                column_names.append(column_name)

        if self.retrieve_dissimilarity_scores:
            column_name = util.handle_column_name_collision(
                column_names, self.dissimilarity_scores_column_name
            )
            result_columns.append(
                knext.Column(
                    knext.ListType(knext.double()),
                    column_name,
                )
            )

        return result_columns

    def _create_empty_table(
        self, table: knext.Table, vectorstore: VectorstorePortObject
    ) -> knext.Table:
        """Constructs an empty KNIME Table with the correct output columns."""
        output_columns = [
            util.OutputColumn(
                self.retrieved_docs_column_name,
                knext.ListType(knext.string()),
                pa.list_(pa.string()),
            )
        ]

        if self.retrieve_metadata:
            for column_name in vectorstore.spec.metadata_column_names:
                output_columns.append(
                    util.OutputColumn(
                        column_name,
                        knext.ListType(knext.string()),
                        pa.list_(pa.string()),
                    )
                )

        if self.retrieve_dissimilarity_scores:
            output_columns.append(
                util.OutputColumn(
                    self.dissimilarity_scores_column_name,
                    knext.ListType(knext.double()),
                    pa.list_(pa.float64()),
                )
            )

        return util.create_empty_table(table, output_columns)


@knext.node(
    name="Vector Store Data Extractor",
    node_type=knext.NodeType.OTHER,
    icon_path=store_icon,
    category=store_category,
    keywords=[
        "GenAI",
        "Generative AI",
        "Retrieval Augmented Generation",
        "RAG",
        "Vector Store to Table",
    ],
)
@knext.input_port(
    name="Vector Store",
    description="The vector store to extract data from.",
    port_type=vector_store_port_type,
)
@knext.output_table(
    name="Extracted Data", description="The data stored inside of the vector store."
)
class VectorStoreDataExtractor:
    """
    Extracts the documents, embeddings and metadata from a vector store into a table.

    Extracts the documents, embeddings and metadata from a vector store into a table.
    This table can be combined with tables extracted from other vector stores to merge multiple
    vector stores into one by creating a new vector store from the combined table.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    document_column_name = knext.StringParameter(
        "Document column name",
        "Specify the name of the output column holding the documents.",
        "Document",
    )

    embedding_column_name = knext.StringParameter(
        "Embedding column name",
        "Specify the name of the output column holding the embeddings.",
        "Embeddings",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        vector_store_spec: VectorstorePortObjectSpec,
    ) -> knext.Schema:
        vector_store_spec.validate_context(ctx)

        if self.document_column_name == self.embedding_column_name:
            raise knext.InvalidParametersError(
                f"Same name ({self.document_column_name}) used for the document and embedding column. Select unique names."
            )
        columns = [
            knext.Column(knext.string(), self.document_column_name),
            knext.Column(knext.list_(knext.double()), self.embedding_column_name),
        ]
        existing_names = [self.document_column_name, self.embedding_column_name]
        columns += [
            knext.Column(
                knext.string(),
                util.handle_column_name_collision(existing_names, meta_col_name),
            )
            for meta_col_name in vector_store_spec.metadata_column_names
        ]
        return knext.Schema.from_columns(columns)

    def execute(
        self, ctx: knext.ExecutionContext, vector_store: VectorstorePortObject
    ) -> knext.Table:
        import pandas as pd

        documents, embeddings = vector_store.get_documents(ctx)
        df = pd.DataFrame()
        df[self.document_column_name] = pd.Series(
            [doc.page_content for doc in documents]
        )
        df[self.embedding_column_name] = pd.Series(
            [np.array(row) for row in embeddings]
        )
        existing_names = [self.document_column_name, self.embedding_column_name]
        for metadata_name in vector_store.spec.metadata_column_names:
            df[util.handle_column_name_collision(existing_names, metadata_name)] = (
                pd.Series([doc.metadata[metadata_name] for doc in documents])
            )
        return knext.Table.from_pandas(df)
