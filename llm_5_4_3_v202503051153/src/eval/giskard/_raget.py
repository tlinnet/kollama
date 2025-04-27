import knime.extension as knext
import knime.api.schema as ks
import util
from models.base import (
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    chat_model_port_type,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
    LLMPortObject,
    LLMPortObjectSpec,
    llm_port_type,
)

from ._base import (
    tortoise_icon,
    eval_category,
    _get_workflow_schema,
    _get_schema_from_workflow_spec,
    ScannerColumn,
    _validate_prediction_workflow_spec,
    _pick_default_workflow_column,
)
from ._tqdm import override_tqdm_with_ctx


@knext.parameter_group("Data")
class InputDataParameters:
    documents_col = knext.ColumnParameter(
        "Documents column",
        "The column containing the documents in the knowledge base.",
        port_index=2,
        column_filter=lambda c: c.ktype == knext.string(),
    )

    embeddings_col = knext.ColumnParameter(
        "Embeddings column",
        "The column containing the embeddings in the knowledge base.",
        port_index=2,
        column_filter=lambda c: c.ktype == knext.ListType(knext.double()),
    )


@knext.parameter_group("Test Set")
class TestSetParameters:
    description = knext.StringParameter(
        "RAG system description",
        """A brief description of the RAG system to be evaluated. This information will be 
        used to generate appropriate test questions tailored to the system.""",
    )

    n_questions = knext.IntParameter(
        "Number of questions",
        "The number of test questions to generate.",
        120,
        min_value=1,
    )


@knext.node(
    "Giskard RAGET Test Set Generator",
    knext.NodeType.OTHER,
    tortoise_icon,
    category=eval_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Model evaluation",
        "Text generation",
        "Large Language Model",
        "Chat Model",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "Chat Model",
    """A configured Chat Model used to generate questions from the provided knowledge base.""",
    chat_model_port_type,
)
@knext.input_port(
    "Embedding Model",
    """A configured Embedding Model used to embed queries to find related documents.""",
    embeddings_model_port_type,
)
@knext.input_table(
    "Knowledge Base",
    """A table containing documents and their embedded representations that are utilized by the RAG system.""",
)
@knext.output_table("Test Set", "The generated test set.")
class TestSetGenerator:
    """
    Generates a test set for evaluating a RAG system.

    This node utilizes the provided knowledge base and the task description of the RAG system
    to automatically generate a diverse set of test questions that are designed to assess
    the performance of various components of the RAG system, such as the retriever,
    generator, and knowledge base quality.
    The questions target specific aspects of the RAG system, helping to identify potential
    weaknesses and areas for improvement.

    The different types of questions include

    - *Simple questions*: Simple questions generated from an excerpt of the knowledge base.
    - *Complex questions*: More complex questions that use paraphrasing.
    - *Distracting questions*: Questions containing distracting information to test the retrieval part of the RAG system.
    - *Situational questions*: Questions that include context information to evaluate if the system can produce answers relevant to the context.
    - *Double questions*: Questions consisting of two separate parts.

    This node does not support Giskard's conversational questions.

    For more details and examples refer to the
    [Giskard documentation](https://docs.giskard.ai/en/stable/open_source/testset_generation/testset_generation/index.html#what-does-raget-do-exactly).

    The output table will contain the following columns:

    - *Question*: The generated question.
    - *Reference Answer*: A reference answer to the question.
    - *Reference Context*: The context used to create the reference answer.
    - *Metadata*: Additional information specific to the type of question.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the chat model or embeddings connector node(s),
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    input_data = InputDataParameters()
    test_set_params = TestSetParameters()

    name_mapping = {
        "id": "ID",
        "question": "Question",
        "reference_answer": "Reference Answer",
        "reference_context": "Reference Context",
        "metadata": "Metadata",
    }

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        chat_model_spec: ChatModelPortObjectSpec,
        embed_model_spec: EmbeddingsPortObjectSpec,
        table_spec: knext.Schema,
    ) -> knext.Schema:
        chat_model_spec.validate_context(ctx)
        embed_model_spec.validate_context(ctx)

        self._check_column_types_exist(table_spec)
        self._set_default_columns(table_spec)

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Question"),
                knext.Column(knext.string(), "Reference Answer"),
                knext.Column(knext.string(), "Reference Context"),
                knext.Column(knext.logical(dict), "Metadata"),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        chat_model_port_object: ChatModelPortObject,
        embed_model_port_object: EmbeddingsPortObject,
        input_table: knext.Table,
    ) -> knext.Table:
        from ._llm_client import KnimeLLMClient
        from giskard.llm.client import set_default_client

        set_default_client(KnimeLLMClient(chat_model_port_object, ctx))

        df = input_table.to_pandas()

        # expensive imports are done here to avoid unnecessary imports in the configure method
        from ._knowledge_base import _KnowledgeBase
        from giskard.rag.testset_generation import generate_testset
        from giskard.rag.question_generators import (
            simple_questions,
            complex_questions,
            distracting_questions,
            situational_questions,
            double_questions,
        )

        kb = _KnowledgeBase(
            data=df,
            documents_col=self.input_data.documents_col,
            embeddings_col=self.input_data.embeddings_col,
            embeddings_model=embed_model_port_object.create_model(ctx),
        )

        with override_tqdm_with_ctx(ctx):
            testset = generate_testset(
                knowledge_base=kb,
                num_questions=self.test_set_params.n_questions,
                agent_description=self.test_set_params.description,
                question_generators=[
                    simple_questions,
                    complex_questions,
                    distracting_questions,
                    situational_questions,
                    double_questions,
                ],
            ).to_pandas()

        # Remove conversation history as we currently do not support conversational agents
        testset = testset.drop("conversation_history", axis=1)

        testset = testset.reset_index()
        testset = testset.rename(columns=self.name_mapping)
        testset = testset.drop("ID", axis=1)

        return knext.Table.from_pandas(testset)

    def _check_column_types_exist(self, table_spec: knext.Schema) -> None:
        has_string_column = False
        has_vector_column = False

        for col in table_spec:
            if col.ktype == knext.string():
                has_string_column = True
            elif col.ktype == knext.ListType(knext.double()):
                has_vector_column = True

            if has_string_column and has_vector_column:
                break

        if not has_string_column:
            raise knext.InvalidParametersError(
                "The knowledge base must contain at least one string column."
            )

        if not has_vector_column:
            raise knext.InvalidParametersError(
                "The knowledge base must contain at least one vector column. A list of doubles is expected."
            )

    def _set_default_columns(self, table_spec: knext.Schema) -> None:
        if not self.input_data.documents_col:
            self.input_data.documents_col = util.pick_default_column(
                table_spec, knext.string()
            )

        if not self.input_data.embeddings_col:
            self.input_data.embeddings_col = util.pick_default_column(
                table_spec, knext.ListType(knext.double())
            )


@knext.node(
    "Giskard RAGET Evaluator",
    knext.NodeType.OTHER,
    tortoise_icon,
    category=eval_category,
    keywords=[
        "Model evaluation",
        "GenAI",
        "Generative AI",
        "LLM",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "LLM or Chat Model", "A large language model or chat model.", llm_port_type
)
@knext.input_port(
    "RAG Workflow",
    "The prediction workflow to analyze with Giskard.",
    knext.PortType.WORKFLOW,
)
@knext.input_table(
    "Test Set", "A test set table generated by the Giskard RAGET Test Set Generator."
)
@knext.output_table("Giskard report data", "The Giskard report as table.")
@knext.output_view("Giskard report", "The Giskard report as HTML.")
class GiskardRAGETEvaluator:
    """
    Evaluates a RAG system with Giskard.

    This node allows to identify potential weaknesses and areas for improvement in a RAG system provided as a
    workflow by evaluating the correctness of the answers. An LLM is used to compare the workflow's answers to the
    reference answers of the test set. The test set can be generated with the
    **Giskard RAGET Test Set Generator** node.

    For more details see the [Giskard documentation](https://docs.giskard.ai/en/stable/open_source/testset_generation/rag_evaluation/).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM connector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "The column in the input table of the workflow that represents the questions for the RAG system.",
        schema_provider=lambda ctx: _get_workflow_schema(ctx, 1, True),
        column_filter=lambda column: column.ktype == knext.string(),
    )

    response_column = knext.ColumnParameter(
        "Response column",
        "The column in the output table of the workflow that represents the LLM responses.",
        schema_provider=lambda ctx: _get_workflow_schema(ctx, 1, False),
        column_filter=lambda column: column.ktype == knext.string(),
    )

    name_mapping = {
        **TestSetGenerator().name_mapping,
        "agent_answer": "Agent Answer",
        "correctness": "Correctness",
        "correctness_reason": "Correctness Reason",
    }

    @property
    def output_columns(self):
        import pandas as pd

        return [
            ScannerColumn("Question", knext.string(), pd.StringDtype()),
            ScannerColumn("Reference Answer", knext.string(), pd.StringDtype()),
            ScannerColumn("Reference Context", knext.string(), pd.StringDtype()),
            ScannerColumn(
                "Metadata",
                knext.logical(dict),
                ks.logical(dict).to_pandas(),
            ),
            ScannerColumn("Agent Answer", knext.string(), pd.StringDtype()),
            ScannerColumn("Correctness", knext.bool_(), pd.BooleanDtype()),
            ScannerColumn("Correctness Reason", knext.string(), pd.StringDtype()),
        ]

    def configure(
        self,
        ctx,
        llm_spec: LLMPortObjectSpec,
        rag_workflow_spec,
        test_set_spec: knext.Schema,
    ) -> knext.Schema:
        llm_spec.validate_context(ctx)
        _validate_prediction_workflow_spec(rag_workflow_spec)
        self._validate_test_set(test_set_spec)

        if not self.prompt_column:
            self.prompt_column = _pick_default_workflow_column(rag_workflow_spec, True)
        if not self.response_column:
            self.response_column = _pick_default_workflow_column(
                rag_workflow_spec, False
            )

        self._validate_selected_params(rag_workflow_spec)

        return knext.Schema.from_columns(
            [
                knext.Column(
                    col.knime_type,
                    col.name,
                )
                for col in self.output_columns
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm: LLMPortObject,
        rag_workflow,
        test_set: knext.Table,
    ):
        from ._llm_client import KnimeLLMClient
        from giskard.llm.client import set_default_client
        import pandas as pd

        input_key = next(iter(rag_workflow.spec.inputs))

        set_default_client(KnimeLLMClient(llm, ctx))

        testset_df = test_set.to_pandas()

        # defined here to be accessed by get_answer_llm for progress reporting
        question_number = 0
        total_questions = testset_df.shape[0]

        # raise error if test set is empty
        if total_questions == 0:
            raise RuntimeError(
                "Execution failed because the provided test set is empty."
            )

        # wraps LLM models
        def get_answer_llm(question: str) -> str:
            if ctx.is_canceled():
                raise RuntimeError("Execution canceled.")

            nonlocal question_number
            nonlocal total_questions

            ctx.set_progress(question_number / total_questions)

            df = pd.DataFrame({self.prompt_column: [question]})
            table = knext.Table.from_pandas(df)

            outputs, _ = rag_workflow.execute({input_key: table})
            answer = outputs[0][self.response_column].to_pandas()
            answer = answer[self.response_column].iloc[0]

            question_number = question_number + 1

            return answer

        testset = self._dataframe_to_testset(testset_df)

        from giskard.rag import evaluate

        report = evaluate(
            get_answer_llm,
            testset=testset,
        )

        html_report = report.to_html()
        html_report = self._remove_empty_html_sections(html_report)
        # The unicode character is not displayed on some Windows machines
        html_report = html_report.replace("\xa0", "&nbsp;")

        report_table = self._report_to_knime_table(report)

        return (
            report_table,
            knext.view_html(html=html_report),
        )

    def _validate_test_set(self, test_set: knext.Schema):
        expected_columns = self.output_columns[0:4]

        # Check if all expected columns exist in the test set with correct types
        for col in expected_columns:
            if col.name not in test_set.column_names:
                raise knext.InvalidParametersError(
                    f"The column '{col.name}' is missing in the test set table."
                )
            expected_type = col.knime_type
            ktype = test_set[col.name].ktype
            if ktype != expected_type:
                raise knext.InvalidParametersError(
                    f"The column '{col.name}' is of type {str(ktype)} but should be of type {str(expected_type)}."
                )

        # Validate that the test set does not contain any other columns
        if not set(test_set.column_names).issubset(
            [col.name for col in expected_columns]
        ):
            raise knext.InvalidParametersError(
                "The test set table contains unexpected columns."
            )

    def _validate_selected_params(self, workflow_spec) -> None:
        util.check_column(
            _get_schema_from_workflow_spec(workflow_spec, return_input_schema=True),
            self.prompt_column,
            knext.string(),
            "prompt",
            "workflow input table",
        )
        util.check_column(
            _get_schema_from_workflow_spec(workflow_spec, return_input_schema=False),
            self.response_column,
            knext.string(),
            "response",
            "workflow output table",
        )

    def _remove_empty_html_sections(self, html_report: str) -> str:
        """If the evaluation is done with no knowledge base or metrics specified, the corresponding section in
        the HTML report is shown but shows None or is empty. This function removes both sections from the report.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_report, "html.parser")

        for section in soup.find_all("div", class_="section-container"):
            section_title = section.find("div", class_="section-title")
            if section_title and section_title.text.strip() in [
                "SELECTED METRICS",
                "KNOWLEDGE BASE OVERVIEW",
            ]:
                section.decompose()

        return soup.decode()

    def _dataframe_to_testset(self, df):
        """Transforms the DataFrame obtained from the input table into a QATestset."""
        # Rename columns
        df = df.rename(columns={v: k for k, v in self.name_mapping.items()})

        # Create Conversation History column as the Test Set Generator currently excludes it
        df["conversation_history"] = [[] for _ in range(len(df))]

        # Insert id column
        df = df.reset_index()
        df = df.rename(columns={"<RowID>": "id"})

        from giskard.rag.testset import QATestset

        return QATestset.from_pandas(df)

    def _report_to_knime_table(self, report) -> knext.Table:
        """Creates a KNIME Table from the report DataFrame."""
        df = report.to_pandas()

        # drop conversation_history as it is currently always empty
        df = df.drop("conversation_history", axis=1)

        df = df.reset_index()
        df = df.rename(columns=self.name_mapping)
        df = df.drop("ID", axis=1)

        for col in self.output_columns:
            df[col.name] = df[col.name].astype(col.pd_type)

        return knext.Table.from_pandas(df)
