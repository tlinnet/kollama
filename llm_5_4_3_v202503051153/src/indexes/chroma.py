import uuid
import numpy as np
import knime.extension as knext
from typing import Any, Optional

from knime.extension import ExecutionContext
from models.base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from .base import (
    BaseVectorStoreCreator,
    VectorstorePortObjectSpec,
    VectorstorePortObject,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
)


chroma_icon = "icons/chroma.png"
chroma_category = knext.category(
    path=store_category,
    level_id="chroma",
    name="Chroma",
    description="Contains nodes for working with Chroma vector stores.",
    icon=chroma_icon,
)

# Keeps the Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME value consistent for backwards compatibility
default_collection_name = "langchain"


def _extract_data_from_chroma_03(persist_directory: str, collection_name: str):
    """Extracts data from vector stores created with Chroma 0.3."""
    from langchain_core.documents import Document
    import json
    import pandas as pd
    import os

    collections_df = pd.read_parquet(
        persist_directory + os.sep + "chroma-collections.parquet",
        engine="pyarrow",
    )
    embeddings_df = pd.read_parquet(
        persist_directory + os.sep + "chroma-embeddings.parquet",
        engine="pyarrow",
    )
    if collection_name not in collections_df["name"].values:
        raise knext.InvalidParametersError(
            f"No Chroma collection named '{collection_name}' was found in the specified directory."
        )
    collection_id = collections_df.loc[collections_df["name"] == collection_name][
        "uuid"
    ].values[0]
    relevant_data = embeddings_df.loc[embeddings_df["collection_uuid"] == collection_id]
    documents = [
        Document(page_content=doc, metadata=metadata)
        for doc, metadata in zip(
            relevant_data["document"],
            [json.loads(d) for d in relevant_data["metadata"]],
        )
    ]
    return documents, np.array(relevant_data["embedding"])


def create_chroma_from_documents(
    collection_name: str,
    embedding_function,
    documents,
    embeddings,
):
    """Creates a langchain_chroma.Chroma instance and populates it with extracted data."""
    from langchain_chroma import Chroma

    db = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    if documents[0].metadata:
        db._collection.add(
            documents=[doc.page_content for doc in documents],
            embeddings=[arr.tolist() for arr in embeddings],
            ids=[str(uuid.uuid4()) for _ in range(len(documents))],
            metadatas=[doc.metadata for doc in documents],
        )
    else:
        db._collection.add(
            documents=[doc.page_content for doc in documents],
            embeddings=[arr.tolist() for arr in embeddings],
            ids=[str(uuid.uuid4()) for _ in range(len(documents))],
        )
    return db


class ChromaVectorstorePortObjectSpec(VectorstorePortObjectSpec):
    """Super type of the spec of local and remote Chroma vector stores."""


class LocalChromaVectorstorePortObjectSpec(
    ChromaVectorstorePortObjectSpec, FilestoreVectorstorePortObjectSpec
):
    """Spec for local chroma instances run within this process."""

    def __init__(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: Optional[list[str]] = None,
        collection_name: str = default_collection_name,
    ) -> None:
        super().__init__(embeddings_spec, metadata_column_names)
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def serialize(self) -> dict:
        data = super().serialize()
        data["collection_name"] = self.collection_name
        return data

    @classmethod
    def deserialize(cls, data: dict):
        super_cls = super().deserialize(data=data)
        return cls(
            super_cls.embeddings_spec,
            super_cls.metadata_column_names,
            data.get("collection_name", default_collection_name),
        )


class ChromaVectorstorePortObject(VectorstorePortObject):
    """Super type of Chroma vector stores"""


chroma_vector_store_port_type = knext.port_type(
    "Chroma Vector Store", ChromaVectorstorePortObject, ChromaVectorstorePortObjectSpec
)


class LocalChromaVectorstorePortObject(
    ChromaVectorstorePortObject, FilestoreVectorstorePortObject
):
    def __init__(
        self,
        spec: ChromaVectorstorePortObjectSpec,
        embeddings_port_object: EmbeddingsPortObject,
        folder_path: Optional[str] = None,
        vectorstore: Optional[Any] = None,
    ) -> None:
        super().__init__(spec, embeddings_port_object, folder_path, vectorstore)

    def save_vectorstore(self, vectorstore_folder: str, vectorstore):
        from langchain_chroma import Chroma

        vectorstore: Chroma = vectorstore
        if (
            vectorstore._persist_directory is None
            or not vectorstore._persist_directory == vectorstore_folder
        ):
            # HACK because Chroma doesn't allow to add or change a persist directory after the fact
            import chromadb
            import chromadb.config

            settings = chromadb.config.Settings(
                is_persistent=True, persist_directory=vectorstore_folder
            )
            existing_collection = vectorstore._collection
            client = chromadb.Client(settings)
            new_collection = client.get_or_create_collection(
                name=existing_collection.name,
                metadata=existing_collection.metadata,
                embedding_function=existing_collection._embedding_function,
            )
            existing_entries = existing_collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            if "data" in existing_entries.keys():
                del existing_entries["data"]
            if "included" in existing_entries.keys():
                del existing_entries["included"]

            try:
                new_collection.add(**existing_entries)
            except ValueError:  # Collection.add raises an Error for empty lists
                raise knext.InvalidParametersError("The vector store is empty.")
            vectorstore = Chroma(
                embedding_function=vectorstore._embedding_function,
                persist_directory=vectorstore_folder,
                client_settings=settings,
                collection_metadata=existing_collection.metadata,
                client=client,
            )

    def load_vectorstore(self, embeddings, vectorstore_path, ctx):
        from langchain_chroma import Chroma

        # if the vectorstore is outdated, we create a new temporary chroma vectorstore in memory
        if self._vectorstore_is_outdated(vectorstore_path):
            extracted_documents, extracted_embeddings = _extract_data_from_chroma_03(
                self._vectorstore_path, self.spec.collection_name
            )
            db = create_chroma_from_documents(
                self.spec.collection_name,
                embeddings,
                extracted_documents,
                extracted_embeddings,
            )
            ctx.set_warning(
                "The vectorstore is outdated. Consider migrating the vectorstore to avoid creating temporary "
                "copies. The node description of the Chroma Vector Store Creator explains the migration."
            )
            return db

        return Chroma(
            collection_name=self.spec.collection_name,
            embedding_function=embeddings,
            persist_directory=vectorstore_path,
        )

    def get_documents(self, ctx: knext.ExecutionContext) -> tuple[list, np.ndarray]:
        from langchain_core.documents import Document
        from langchain_chroma import Chroma

        store: Chroma = self.load_store(ctx)
        content = store.get(include=["embeddings", "metadatas", "documents"])
        if content["metadatas"][0] is None:
            documents = [Document(page_content=doc) for doc in content["documents"]]
        else:
            documents = [
                Document(page_content=doc, metadata=metadata)
                for doc, metadata in zip(content["documents"], content["metadatas"])
            ]
        return documents, np.array(content["embeddings"])

    def get_metadata_filter(self, filter_parameter) -> dict:
        # Chroma expects a filter dictionary with multiple key-value pairs to be combined with an AND operator
        # reference: https://docs.trychroma.com/guides#using-logical-operators
        if not filter_parameter:
            return {}

        if len(filter_parameter) == 1:
            return {
                filter_parameter[0].metadata_column: filter_parameter[0].metadata_value
            }

        filter_list = [
            {group.metadata_column: group.metadata_value} for group in filter_parameter
        ]

        return {"$and": filter_list}

    def _vectorstore_is_outdated(self, path: str):
        import os

        path = path + os.sep + "chroma-collections.parquet"
        return os.path.isfile(path)


local_chroma_vector_store_port_type = knext.port_type(
    "Chroma Vector Store",
    LocalChromaVectorstorePortObject,
    LocalChromaVectorstorePortObjectSpec,
)


@knext.node(
    "Chroma Vector Store Creator",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.input_table(
    name="Documents",
    description="""Table containing a string column representing documents that will be used in the vector store.""",
)
@knext.output_port(
    "Chroma Vector Store",
    "The created Chroma vector store.",
    local_chroma_vector_store_port_type,
)
class ChromaVectorStoreCreator(BaseVectorStoreCreator):
    """
    Creates a Chroma vector store from a string column and an embeddings model.

    The node generates a Chroma vector store that uses the given embeddings model to map documents to a numerical vector that captures
    the semantic meaning of the document.

    By default, the node embeds the selected documents using the embeddings model, but it is also possible to create the vector store
    from existing embeddings by specifying the corresponding embeddings column in the node dialog.

    Downstream nodes, such as the **Vector Store Retriever**, utilize the vector store to find documents with similar
    semantic meaning when given a query.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.

    **Note**: Chroma changed their data layout which requires downstream nodes to create temporary copies when
    reading an outdated vector store. To migrate an outdated vector store, the **Vector Store Data Extractor**
    can be used to extract its data, which can then be used to create a new vectorstore.
    """

    # redefined here to enforce the parameter order in the dialog
    document_column = BaseVectorStoreCreator.document_column

    embeddings_column = BaseVectorStoreCreator.embeddings_column

    collection_name = knext.StringParameter(
        "Collection name",
        "Specify the collection name of the vector store.",
        default_collection_name,
        since_version="5.3.2",
    )

    def _configure(
        self,
        embeddings_spec: EmbeddingsPortObjectSpec,
        metadata_column_names: list[str],
    ) -> LocalChromaVectorstorePortObjectSpec:
        if self.collection_name == "":
            raise knext.InvalidParametersError("The collection name must not be empty.")
        return LocalChromaVectorstorePortObjectSpec(
            embeddings_spec=embeddings_spec,
            metadata_column_names=metadata_column_names,
            collection_name=self.collection_name,
        )

    def _create_port_object(
        self,
        ctx: ExecutionContext,
        embeddings_obj: EmbeddingsPortObject,
        documents: list,
        metadata_column_names: list[str],
        embeddings,
    ) -> LocalChromaVectorstorePortObject:
        from langchain_chroma import Chroma

        embeddings_model = embeddings_obj.create_model(ctx)
        if embeddings is None:
            db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings_model,
                collection_name=self.collection_name,
            )
        else:
            db = create_chroma_from_documents(
                self.collection_name, embeddings_model, documents, embeddings
            )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(
                embeddings_spec=embeddings_obj.spec,
                metadata_column_names=metadata_column_names,
                collection_name=self.collection_name,
            ),
            embeddings_port_object=embeddings_obj,
            vectorstore=db,
        )


@knext.node(
    "Chroma Vector Store Reader",
    knext.NodeType.SOURCE,
    chroma_icon,
    category=chroma_category,
    keywords=[
        "RAG",
        "Retrieval Augmented Generation",
        "Embeddings",
    ],
)
@knext.input_port(
    "Embeddings",
    "The embeddings model to use for the vector store.",
    embeddings_model_port_type,
)
@knext.output_port(
    "Chroma Vector Store",
    "The loaded vector store.",
    local_chroma_vector_store_port_type,
)
class ChromaVectorStoreReader:
    """
    Reads a Chroma vector store created with LangChain from a local path.

    This node allows you to read a Chroma vector store created with [LangChain](https://python.langchain.com/docs/integrations/vectorstores/chroma)
    from a local path. If you want to create a new vector store, use the **Chroma Vector Store Creator** instead.

    A *vector store* is a data structure or storage mechanism that stores a collection of numerical vectors
    along with their corresponding documents. The vector store enables efficient storage, retrieval, and similarity
    search operations on these vectors and their associated data.

    If the vector store was created with LangChain in Python, the embeddings model is not stored with the vectorstore, so it has to be provided separately via the matching **Embeddings Model Connector** node.

    On execution, the node will extract a document from the store to obtain information about the document's metadata. This assumes that each document in the vector store has the same kind of metadata attached to it.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    persist_directory = knext.StringParameter(
        "Vector store directory",
        """The local directory in which the vector store is stored.""",
    )

    collection_name = knext.StringParameter(
        "Collection name",
        "Specify the collection name of the vector store.",
        default_collection_name,
        since_version="5.3.2",
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        embeddings_spec: EmbeddingsPortObjectSpec,
    ) -> LocalChromaVectorstorePortObjectSpec:
        embeddings_spec.validate_context(ctx)
        if not self.persist_directory:
            raise knext.InvalidParametersError("No vector store directory specified.")

        if self.collection_name == "":
            raise knext.InvalidParametersError("The collection name must not be empty.")

        return LocalChromaVectorstorePortObjectSpec(
            embeddings_spec, collection_name=self.collection_name
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        embeddings_port_object: EmbeddingsPortObject,
    ) -> ChromaVectorstorePortObject:
        version = self._get_database_version(self.persist_directory)
        chroma = self._create_chroma_with_version(version, embeddings_port_object, ctx)

        document_list = chroma.similarity_search("a", k=1)
        metadata_keys = (
            [key for key in document_list[0].metadata] if len(document_list) > 0 else []
        )

        return LocalChromaVectorstorePortObject(
            LocalChromaVectorstorePortObjectSpec(
                embeddings_port_object.spec, metadata_keys, self.collection_name
            ),
            embeddings_port_object,
            vectorstore=chroma,
        )

    def _get_database_version(self, persist_directory: str) -> str:
        """Returns the Chroma version used to create the vector store.
        Raises an error if the specified directory or collection name is wrong."""
        import os

        # check if sqlite3 file (new data layout) exists
        path = persist_directory + os.sep + "chroma.sqlite3"
        if not os.path.isfile(path):
            if os.path.isfile(
                persist_directory + os.sep + "chroma-collections.parquet"
            ):
                return "0.3"
            else:
                raise knext.InvalidParametersError(
                    "The specified directory does not contain a Chroma vector store."
                )
        try:
            import sqlite3

            sqliteConnection = sqlite3.connect(path)
            sql_query = """SELECT * FROM collections;"""
            cursor = sqliteConnection.cursor()
            cursor.execute(sql_query)

            # check if collection name exists
            column_names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            column_idx = column_names.index("name")
            if self.collection_name not in [row[column_idx] for row in rows]:
                raise knext.InvalidParametersError(
                    f"No Chroma collection named '{self.collection_name}' was found in the specified directory."
                )

            # check if topic column exists, as it indicates the store was created with Chroma 0.4
            if "topic" in column_names:
                return "0.4"
            else:
                return "0.5"
        except sqlite3.Error:
            raise RuntimeError(
                "Failed to connect to the vector store. This may be because the vector store has "
                "an unexpected schema."
            )
        finally:
            if sqliteConnection:
                sqliteConnection.close()

    def _create_chroma_with_version(
        self, version: str, embeddings_port_object: EmbeddingsPortObject, ctx
    ):
        """Returns a langchain_chroma.Chroma instance populated with the data of the specified vectorstore.
        For outdated vectorstores, the data is manually extracted."""
        from langchain_chroma import Chroma

        if version == "0.5":
            return Chroma(
                self.collection_name,
                embeddings_port_object.create_model(ctx),
                self.persist_directory,
            )
        elif version == "0.3":
            extracted_documents, extracted_embeddings = _extract_data_from_chroma_03(
                self.persist_directory, self.collection_name
            )
        elif version == "0.4":
            extracted_documents, extracted_embeddings = (
                self._extract_data_from_chroma_04(embeddings_port_object, ctx)
            )
        else:
            raise NotImplementedError("Encountered invalid Chroma version.")
        db = create_chroma_from_documents(
            self.collection_name,
            embeddings_port_object.create_model(ctx),
            extracted_documents,
            extracted_embeddings,
        )
        return db

    def _extract_data_from_chroma_04(
        self, embeddings_port_object: EmbeddingsPortObject, ctx
    ):
        """Chroma auto-migrates databases from version 0.4 when reading with version 0.5, so we copy the
        directory to a temporary directory and extract the data from there."""
        from tempfile import TemporaryDirectory
        import shutil
        import os
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        with TemporaryDirectory() as temp_dir:
            try:
                shutil.copytree(
                    self.persist_directory, os.path.join(temp_dir, "vectorstore")
                )
                temp_dir_path = os.path.join(temp_dir, "vectorstore")
                chroma = Chroma(
                    self.collection_name,
                    embeddings_port_object.create_model(ctx),
                    temp_dir_path,
                )
                content = chroma.get(include=["embeddings", "metadatas", "documents"])
                if content["metadatas"][0] is None:
                    documents = [
                        Document(page_content=doc) for doc in content["documents"]
                    ]
                else:
                    documents = [
                        Document(page_content=doc, metadata=metadata)
                        for doc, metadata in zip(
                            content["documents"], content["metadatas"]
                        )
                    ]
            finally:
                # HACK because Chroma does not close its connection to the vector store, so the temp dir can't be removed on Windows
                chroma._client._system.stop()
                del chroma
        return documents, np.array(content["embeddings"])
