import knime.extension as knext
import util


from models.base import LLMPortObjectSpec, LLMPortObject, ChatConversationSettings
from tools.base import (
    tool_list_port_type,
    ToolListPortObject,
    ToolListPortObjectSpec,
)

from knime.extension.nodes import (
    get_port_type_for_id,
    get_port_type_for_spec_type,
    load_port_object,
    save_port_object,
    FilestorePortObject,
)
from base import AIPortObjectSpec

import os


agent_icon = "icons/agent.png"
agent_category = knext.category(
    path=util.main_category,
    level_id="agents",
    name="Agents",
    description="",
    icon=agent_icon,
)


@knext.parameter_group(label="Credentials")
class CredentialsSettings:
    credentials_param = knext.StringParameter(
        label="Credentials parameter",
        description="Credentials parameter name for accessing Google Search API key.",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )


@knext.parameter_group(label="Prompt Settings")
class ChatMessageSettings:
    message = knext.MultilineStringParameter(
        "New message", "Specify the new message to be sent to the agent."
    )


# TODO: Add agent type in the future?
class AgentPortObjectSpec(AIPortObjectSpec):
    def __init__(
        self,
        llm_spec: LLMPortObjectSpec,
    ) -> None:
        self._llm_spec = llm_spec
        self._llm_type = get_port_type_for_spec_type(type(llm_spec))

    @property
    def llm_spec(self) -> LLMPortObjectSpec:
        return self._llm_spec

    @property
    def llm_type(self) -> knext.PortType:
        return self._llm_type

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._llm_spec.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "llm": {"type": self.llm_type.id, "spec": self.llm_spec.serialize()},
        }

    @classmethod
    def deserialize_llm_spec(cls, data: dict):
        llm_data = data["llm"]
        spec_type = get_port_type_for_id(llm_data["type"])

        return spec_type.spec_class.deserialize(llm_data["spec"])

    @classmethod
    def deserialize(cls, data: dict) -> "AgentPortObjectSpec":
        return cls(cls.deserialize_llm_spec(data))


class AgentPortObject(FilestorePortObject):
    def __init__(self, spec: AgentPortObjectSpec, llm: LLMPortObject) -> None:
        super().__init__(spec)
        self._llm = llm

    @property
    def llm(self) -> LLMPortObject:
        return self._llm

    def create_agent(self, ctx, tools):
        raise NotImplementedError()

    def write_to(self, file_path):
        os.makedirs(file_path)
        llm_path = os.path.join(file_path, "llm")
        save_port_object(self.llm, llm_path)

    @classmethod
    def read_from(cls, spec: AgentPortObjectSpec, file_path: str) -> "AgentPortObject":
        llm_path = os.path.join(file_path, "llm")
        llm = load_port_object(spec.llm_type.object_class, spec.llm_spec, llm_path)
        return cls(spec, llm)


agent_port_type = knext.port_type("Agent", AgentPortObject, AgentPortObjectSpec)


@knext.node(
    "Agent Prompter",
    knext.NodeType.PREDICTOR,
    agent_icon,
    category=agent_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
        "OpenAI",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port("Agent", "The agent to prompt.", agent_port_type)
@knext.input_port("Tools", "The tools the agent can use.", tool_list_port_type)
@knext.input_table(
    "Conversation",
    """Table containing the conversation history. 
                    Has to contain at least two string columns, which can be empty if this is the beginning of the conversation.
                    """,
)
@knext.output_table(
    "Continued conversation",
    """The conversation table extended by two rows.
                    One row for the user's prompt and one row for the agent's response.
                    Can be used as input in subsequent Agent Prompter executions to continue the conversation further.
                    """,
)
class AgentPrompter:
    """
    Combines an agent with a set of tools and prompts it with a user-provided prompt.

    This node supplies an LLM agent with a set of tools and the conversation history, and
    prompts it with the user-provided query.

    The conversation table is expected to have at least two string columns that define previous conversation.
    If this is the start of the conversation, the conversation table can be empty.

    The agent always receives the full conversation table as context, which can lead to
    slower execution times for longer conversations, or execution failures if the context
    becomes too large for the LLM. If you experience such issues, you can truncate the conversation
    table by only keeping the last few messages, or use an LLM to summarize the conversation held so far.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable was not saved and will therefore not be available to downstream nodes.
    """

    conversation_settings = ChatConversationSettings(port_index=2)
    message_settings = ChatMessageSettings()
    enable_debug_output = knext.BoolParameter(
        "Enable debug output",
        "If checked, prints the output of LangChain's debug mode to the console.",
        False,
        is_advanced=True,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        agent_spec: AgentPortObjectSpec,
        tools_spec: ToolListPortObjectSpec,
        input_table_spec: knext.Schema,
    ):
        agent_spec.validate_context(ctx)
        tools_spec.validate_context(ctx)
        self.conversation_settings.configure(input_table_spec)
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), self.conversation_settings.role_column),
                knext.Column(knext.string(), self.conversation_settings.content_column),
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        agent_obj: AgentPortObject,
        tools_obj: ToolListPortObject,
        input_table: knext.Table,
    ):
        chat_history_df = input_table[
            [
                self.conversation_settings.role_column,
                self.conversation_settings.content_column,
            ]
        ].to_pandas()

        # TODO return messages might depend on the type of model (i.e. chat needs it llm doesn't)?

        import langchain
        import langchain.agents

        langchain.debug = self.enable_debug_output

        messages = self.conversation_settings.create_messages(chat_history_df)

        tools = tools_obj.create_tools(ctx)
        agent = agent_obj.create_agent(ctx, tools)

        agent_exec = langchain.agents.AgentExecutor(
            agent=agent,
            tools=tools,
        )

        response = agent_exec.invoke(
            {
                "input": self.message_settings.message,
                "chat_history": messages,
            }
        )

        user_input_row = ["Human", self.message_settings.message]
        agent_output_row = ["AI", response["output"]]

        chat_history_df.loc[f"Row{len(chat_history_df)}"] = user_input_row
        chat_history_df.loc[f"Row{len(chat_history_df)}"] = agent_output_row

        return knext.Table.from_pandas(chat_history_df)
