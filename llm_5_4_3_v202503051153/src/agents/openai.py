from models.openai import (
    OpenAIChatModelPortObject,
    OpenAIChatModelPortObjectSpec,
    openai_chat_port_type,
    openai_icon,
)
import knime.extension as knext
from .base import AgentPortObject, AgentPortObjectSpec
from .base import agent_category


import re


class OpenAIFunctionsAgentPortObjectSpec(AgentPortObjectSpec):
    def __init__(self, llm_spec: OpenAIChatModelPortObjectSpec, system_message) -> None:
        super().__init__(llm_spec)
        self._system_message = system_message

    @property
    def system_message(self) -> str:
        return self._system_message

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_message"] = self.system_message
        return data

    @classmethod
    def deserialize(cls, data) -> "OpenAIFunctionsAgentPortObjectSpec":
        return cls(cls.deserialize_llm_spec(data), data["system_message"])


class OpenAiFunctionsAgentPortObject(AgentPortObject):
    def __init__(
        self, spec: AgentPortObjectSpec, llm: OpenAIChatModelPortObject
    ) -> None:
        super().__init__(spec, llm)

    @property
    def spec(self) -> OpenAIFunctionsAgentPortObjectSpec:
        return super().spec

    def validate_tools(self, tools):
        pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        for tool in tools:
            if not pattern.match(tool.name):
                raise knext.InvalidParametersError(
                    f"Invalid tool name '{tool.name}'. The name must be 1 to 64 characters long and can only contain alphanumeric characters, underscores, and hyphens."
                )
            if not tool.description or not tool.description.strip():
                raise knext.InvalidParametersError(
                    f"Invalid or missing tool description for tool: {tool.name}."
                )

        return tools

    def create_agent(self, ctx, tools):
        from langchain_core.prompts import MessagesPlaceholder
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import create_openai_functions_agent

        llm = self.llm.create_model(ctx)
        tools = self.validate_tools(tools)
        prompt = ChatPromptTemplate(
            [
                ("system", self.spec.system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        return create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )


openai_functions_agent_port_type = knext.port_type(
    "OpenAI Functions Agent",
    OpenAiFunctionsAgentPortObject,
    OpenAIFunctionsAgentPortObjectSpec,
)


@knext.node(
    "OpenAI Functions Agent Creator",
    knext.NodeType.SOURCE,
    openai_icon,
    agent_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI", "Azure"],
)
@knext.input_port(
    "(Azure) OpenAI Chat Model",
    "The (Azure) OpenAI chat model used by the agent to make decisions.",
    openai_chat_port_type,
)
@knext.output_port(
    "OpenAI Functions Agent",
    "An agent that can use OpenAI functions.",
    openai_functions_agent_port_type,
)
class OpenAIFunctionsAgentCreator:
    """
    Creates an agent that utilizes the function calling feature of (Azure) OpenAI chat models.

    This node creates an agent based on (Azure) OpenAI chat models that support function calling
    (e.g. the 0613 models) and can be primed with a custom system message.

    The *system message* plays an essential role in defining the behavior of the agent and how it interacts with users and tools.
    Before adjusting other model settings, it is recommended to experiment with the system message first, as it has
    the most significant impact on the behavior of the agent.

    An *agent* is an LLM that is configured to pick a tool from
    a set of tools to best answer the user prompts, when appropriate.

    **For Azure**: Make sure to use the correct API, since function calling is only available since API version
    '2023-07-01-preview'. For more information, check the
    [Microsoft Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python)

    **Note**: These agents do not support tools with whitespaces in their names.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the chat model,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    system_message = knext.MultilineStringParameter(
        "System message",
        """Specify the system message defining the behavior of the agent.""",
        """You are a helpful AI assistant. Never solely rely on your own knowledge, but use tools to get information before answering. """,
    )

    def configure(self, ctx, chat_model_spec: OpenAIChatModelPortObjectSpec):
        chat_model_spec.validate_context(ctx)
        return OpenAIFunctionsAgentPortObjectSpec(chat_model_spec, self.system_message)

    def execute(self, ctx, chat_model: OpenAIChatModelPortObject):
        return OpenAiFunctionsAgentPortObject(
            OpenAIFunctionsAgentPortObjectSpec(chat_model.spec, self.system_message),
            chat_model,
        )
