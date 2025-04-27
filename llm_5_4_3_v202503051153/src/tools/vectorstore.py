from typing import Optional

import knime.extension as knext
from knime.extension.nodes import (
    get_port_type_for_id,
    get_port_type_for_spec_type,
    load_port_object,
    save_port_object,
    FilestorePortObject,
)
from .base import ToolPortObjectSpec, ToolPortObject
from models.base import LLMPortObject, LLMPortObjectSpec, llm_port_type
from indexes.base import (
    VectorstorePortObject,
    VectorstorePortObjectSpec,
    FilestoreVectorstorePortObjectSpec,
    FilestoreVectorstorePortObject,
    store_category,
    vector_store_port_type,
)
from .base import ToolListPortObject, ToolListPortObjectSpec, tool_list_port_type
import os


class VectorToolPortObjectSpec(ToolPortObjectSpec):
    def __init__(self, name, description, top_k, source_metadata) -> None:
        super().__init__(name, description)
        self._top_k = top_k
        self._source_metadata = source_metadata

    @property
    def top_k(self):
        return self._top_k

    @property
    def source_metadata(self) -> Optional[str]:
        return self._source_metadata

    def serialize(self) -> dict:
        return {
            "name": self._name,
            "description": self._description,
            "top_k": self._top_k,
            "source_metadata": self._source_metadata,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            data["name"],
            data["description"],
            data["top_k"],
            data.get("source_metadata"),
        )


class VectorToolPortObject(ToolPortObject):
    def __init__(
        self,
        spec: VectorToolPortObjectSpec,
        llm: LLMPortObject,
        vectorstore: VectorstorePortObject,
    ) -> None:
        super().__init__(spec)
        self._llm = llm
        self._vectorstore = vectorstore

    @property
    def spec(self) -> VectorToolPortObjectSpec:
        return super().spec

    @property
    def llm(self) -> LLMPortObject:
        return self._llm

    @property
    def vectorstore(self) -> VectorstorePortObject:
        return self._vectorstore

    def _create_function(self, ctx, source_metadata: Optional[str] = None):
        """
        This method creates a function based on the given context and retrieval sources flag.
        """
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        llm = self._llm.create_model(ctx)
        vectorstore = self._vectorstore.load_store(ctx)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.spec.top_k})

        if source_metadata:
            from typing_extensions import List, TypedDict, Annotated
            from langchain_core.runnables import RunnableLambda

            prompt_message = (
                "You are an AI assistant for answering questions. You will be provided with the necessary "
                "context to answer the question. Specify the sources of the pieces of retrieved context used "
                "to answer the question. If the context does not contain the answer, just say that you "
                "don't know. Use three sentences maximum and keep the answer concise.\n"
                "Question: {input}\n\nContext: {context}\n\nAnswer:"
            )
            prompt = ChatPromptTemplate(
                [
                    ("human", prompt_message),
                ]
            )

            class AnswerWithSources(TypedDict):
                answer: str
                sources: Annotated[
                    List[str],
                    ...,
                    "List of sources used to answer the question",
                ]

            def retrieve(state: dict):
                retrieved_docs = retriever.invoke(state["input"])
                return {"input": state["input"], "context": retrieved_docs}

            def generate(state: dict):
                docs_content = "\n\n".join(
                    "Content: "
                    + doc.page_content
                    + "\n"
                    + "Source: "
                    + doc.metadata[source_metadata]
                    for doc in state["context"]
                )
                messages = prompt.invoke(
                    {"input": state["input"], "context": docs_content}
                )
                structured_llm = llm.with_structured_output(
                    AnswerWithSources, method="function_calling"
                )
                response = structured_llm.invoke(messages)
                return response

            retrieval_chain = RunnableLambda(retrieve) | RunnableLambda(generate)
            return retrieval_chain

        prompt_message = (
            "You are an AI assistant for answering questions. You will be provided with the necessary "
            "context to answer the question. If the context does not contain the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n"
            "Question: {input}\n\nContext: {context}\n\nAnswer:"
        )
        prompt = ChatPromptTemplate(
            [
                ("human", prompt_message),
            ]
        )
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        return rag_chain

    def create(self, ctx):
        from langchain_core.tools import StructuredTool
        from ._schema import RetrievalQAToolSchema

        retrieval_chain = self._create_function(ctx, self.spec.source_metadata)
        return StructuredTool.from_function(
            name=self.spec.serialize()["name"],
            args_schema=RetrievalQAToolSchema,
            func=lambda input: retrieval_chain.invoke({"input": input}),
            description=self.spec.serialize()["description"],
        )


class FilestoreVectorToolPortObjectSpec(VectorToolPortObjectSpec):
    def __init__(
        self,
        name,
        description,
        top_k,
        source_metadata,
        llm_spec: LLMPortObjectSpec,
        vectorstore_spec: VectorstorePortObjectSpec,
    ) -> None:
        super().__init__(name, description, top_k, source_metadata)
        self._llm_spec = llm_spec
        self._llm_type = get_port_type_for_spec_type(type(llm_spec))
        self._vectorstore_spec = vectorstore_spec
        self._vectorstore_type = get_port_type_for_spec_type(type(vectorstore_spec))

    @property
    def llm_spec(self) -> LLMPortObjectSpec:
        return self._llm_spec

    @property
    def llm_type(self) -> knext.PortType:
        return self._llm_type

    @property
    def vectorstore_spec(self) -> VectorstorePortObjectSpec:
        return self._vectorstore_spec

    @property
    def vectorstore_type(self) -> knext.PortType:
        return self._vectorstore_type

    def validate_context(self, ctx: knext.ConfigurationContext):
        self.llm_spec.validate_context(ctx)
        self.vectorstore_spec.validate_context(ctx)

    def serialize(self) -> dict:
        data = super().serialize()
        data["llm_spec"] = self.llm_spec.serialize()
        data["llm_type"] = self.llm_type.id
        data["vectorstore_spec"] = self.vectorstore_spec.serialize()
        data["vectorstore_type"] = self.vectorstore_type.id
        return data

    @classmethod
    def deserialize(cls, data: dict) -> FilestoreVectorstorePortObject:
        llm_type: knext.PortType = get_port_type_for_id(data["llm_type"])
        llm_spec = llm_type.spec_class.deserialize(data["llm_spec"])
        vectorstore_type: knext.PortType = get_port_type_for_id(
            data["vectorstore_type"]
        )
        vectorstore_spec = vectorstore_type.spec_class.deserialize(
            data["vectorstore_spec"]
        )
        return cls(
            data["name"],
            data["description"],
            data["top_k"],
            data.get("source_metadata"),
            llm_spec,
            vectorstore_spec,
        )


class FilestoreVectorToolPortObject(VectorToolPortObject, FilestorePortObject):
    def __init__(
        self,
        spec: VectorToolPortObjectSpec,
        llm: LLMPortObject,
        vectorstore: VectorstorePortObject,
    ) -> None:
        super().__init__(spec, llm, vectorstore)

    def write_to(self, file_path: str) -> None:
        os.makedirs(file_path)
        llm_path = os.path.join(file_path, "llm")
        save_port_object(self.llm, llm_path)
        vectorstore_path = os.path.join(file_path, "vectorstore")
        save_port_object(self._vectorstore, vectorstore_path)

    @classmethod
    def read_from(
        cls, spec: FilestoreVectorToolPortObjectSpec, file_path: str
    ) -> "FilestoreVectorToolPortObject":
        llm_path = os.path.join(file_path, "llm")
        llm = load_port_object(spec.llm_type.object_class, spec.llm_spec, llm_path)
        vectorstore_path = os.path.join(file_path, "vectorstore")
        vectorstore = load_port_object(
            spec.vectorstore_type.object_class, spec.vectorstore_spec, vectorstore_path
        )
        return cls(spec, llm, vectorstore)


# not actually output by any node but needs to be registered in the framework,
# such that the ToolListPortObject can load FilestoreVectorstorePortObjects via load_port_object
_filestore_vector_tool_port_type = knext.port_type(
    "Filestore Vector Store Tool",
    FilestoreVectorToolPortObject,
    FilestoreVectorToolPortObjectSpec,
)


@knext.node(
    "Vector Store to Tool",
    knext.NodeType.MANIPULATOR,
    icon_path="icons/store.png",
    category=store_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Agent",
        "OpenAI",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port("LLM", "A large language model.", llm_port_type)
@knext.input_port("Vector Store", "A loaded vector store.", vector_store_port_type)
@knext.output_port("Agent Tool", "A tool for an agent to use.", tool_list_port_type)
class VectorStoreToTool:
    """
    Creates an agent tool from a vector store.

    This node turns a vector store into a tool by providing it with a name and a description.
    This tool can then be used by an agent during the execution of the **Agent Prompter** node to dynamically
    retrieve relevant documents from the underlying vector store.

    A meaningful name and description are very important, for example:

    *Name*: KNIME_Node_Description_QA_System

    *Description*: Use this tool whenever you need information about which nodes a user would need in a given
    situation or if you need information about nodes' configuration options.

    **Note**: *OpenAI Functions Agents* require the name to contain no whitespace while other kinds
    of agents may not have this restriction.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    tool_name = knext.StringParameter(
        label="Tool name",
        description="""The name for the tool. Note that *OpenAI Functions Agents* require the name to contain no whitespace, while other types of agents may not have this restriction.""",
    )

    tool_description = knext.StringParameter(
        label="Tool description",
        description="""The description of the tool by which an agent decides whether or not to use the tool. 
        Provide a meaningful description to make the agent decide more optimally. Note that the tool description cannot be empty for *OpenAI Functions Agents*.""",
    )

    top_k = knext.IntParameter(
        label="Retrieved documents",
        description="The number of top results that the tool will provide from the vector store.",
        default_value=5,
        is_advanced=True,
    )

    retrieve_sources = knext.BoolParameter(
        "Retrieve sources from documents",
        "Whether or not to retrieve document sources if provided.",
        default_value=False,
        since_version="5.2.0",
    )

    source_metadata = knext.StringParameter(
        "Source metadata",
        "The metadata containing the sources of the documents.",
        "",
        choices=lambda ctx: (
            specs.metadata_column_names
            if (specs := ctx.get_input_specs()[1])
            and len(specs.metadata_column_names) >= 1
            else []
        ),
        since_version="5.2.0",
    ).rule(knext.OneOf(retrieve_sources, [True]), knext.Effect.SHOW)

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        llm_spec: LLMPortObjectSpec,
        vectorstore_spec: VectorstorePortObjectSpec,
    ) -> ToolListPortObjectSpec:
        tool_spec = self._create_tool_spec(llm_spec, vectorstore_spec)
        tool_spec.validate_context(ctx)
        return ToolListPortObjectSpec([tool_spec])

    def _create_tool_spec(
        self, llm_spec: LLMPortObjectSpec, vectorstore_spec: VectorstorePortObjectSpec
    ) -> FilestoreVectorstorePortObjectSpec:
        if self.retrieve_sources and not self.source_metadata:
            raise knext.InvalidParametersError(
                "Select the metadata that holds the sources."
            )
        if self.retrieve_sources:
            return FilestoreVectorToolPortObjectSpec(
                self.tool_name,
                self.tool_description,
                self.top_k,
                self.source_metadata,
                llm_spec,
                vectorstore_spec,
            )
        else:
            return FilestoreVectorToolPortObjectSpec(
                self.tool_name,
                self.tool_description,
                self.top_k,
                None,
                llm_spec,
                vectorstore_spec,
            )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        vectorstore: VectorstorePortObject,
    ) -> ToolListPortObject:
        tool = FilestoreVectorToolPortObject(
            spec=self._create_tool_spec(llm.spec, vectorstore.spec),
            llm=llm,
            vectorstore=vectorstore,
        )
        return ToolListPortObject(ToolListPortObjectSpec([tool.spec]), [tool])
