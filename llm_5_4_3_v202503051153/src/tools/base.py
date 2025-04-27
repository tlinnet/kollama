import knime.extension as knext
from knime.extension.nodes import (
    FilestorePortObject,
    get_port_type_for_id,
    get_port_type_for_spec_type,
    save_port_object,
    load_port_object,
)
from base import AIPortObjectSpec
from typing import List
import os
from indexes.base import store_category  # TODO add separate tool category?


class ToolPortObjectSpec(AIPortObjectSpec):
    def __init__(self, name, description) -> None:
        super().__init__()
        self._name = name
        self._description = description

    def serialize(self) -> dict:
        return {"name": self._name, "description": self._description}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["name"], data["description"])


class ToolPortObject(knext.PortObject):
    def __init__(self, spec: ToolPortObjectSpec) -> None:
        super().__init__(spec)

    def create(self, ctx):
        raise NotImplementedError()


class ToolListPortObjectSpec(AIPortObjectSpec):
    def __init__(self, tool_specs: List[ToolPortObjectSpec]) -> None:
        self._tool_specs = tool_specs
        self._tool_types = [
            get_port_type_for_spec_type(type(spec)) for spec in tool_specs
        ]

    @property
    def tool_types(self) -> List[knext.PortType]:
        return self._tool_types

    @property
    def tool_specs(self) -> List[ToolPortObjectSpec]:
        return self._tool_specs

    def validate_context(self, ctx: knext.ConfigurationContext):
        for tool_spec in self._tool_specs:
            tool_spec.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "tool_specs": [
                {"port_type": port_type.id, "spec": spec.serialize()}
                for port_type, spec in zip(self.tool_types, self.tool_specs)
            ]
        }

    @classmethod
    def deserialize(cls, data) -> "ToolListPortObjectSpec":
        tool_specs = [
            get_port_type_for_id(spec_data["port_type"]).spec_class.deserialize(
                spec_data["spec"]
            )
            for spec_data in data["tool_specs"]
        ]
        return cls(tool_specs)


class ToolListPortObject(FilestorePortObject):
    def __init__(
        self, spec: ToolListPortObjectSpec, tools: List[ToolPortObject]
    ) -> None:
        super().__init__(spec)
        self._tool_list = tools

    @property
    def spec(self) -> ToolListPortObjectSpec:
        return super().spec

    @property
    def tools(self) -> List[ToolPortObject]:
        return self._tool_list

    def create_tools(self, ctx):
        return [tool.create(ctx) for tool in self.tools]

    def write_to(self, file_path):
        os.makedirs(file_path)
        for i, tool in enumerate(self.tools):
            tool_path = os.path.join(file_path, str(i))
            save_port_object(tool, tool_path)

    @classmethod
    def read_from(
        cls, spec: ToolListPortObjectSpec, file_path: str
    ) -> "ToolListPortObject":
        tools = [
            load_port_object(
                port_type.object_class, tool_spec, os.path.join(file_path, str(i))
            )
            for i, (port_type, tool_spec) in enumerate(
                zip(spec.tool_types, spec.tool_specs)
            )
        ]
        return cls(spec, tools)


tool_list_port_type = knext.port_type(
    "Tool list", ToolListPortObject, ToolListPortObjectSpec
)


@knext.node(
    "Tool Concatenator",
    knext.NodeType.SOURCE,
    icon_path="icons/store.png",
    category=store_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Agent",
        "Functions",
        "OpenAI",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "Agent Tool(s)", "One or more tools for an agent to use.", tool_list_port_type
)
@knext.input_port(
    "Agent Tool(s)", "One or more tools for an agent to use.", tool_list_port_type
)
@knext.output_port(
    "Agent Tools",
    "The concatenated tool list for an agent to use.",
    tool_list_port_type,
)
class ToolCombiner:
    """
    Concatenates two lists of LLM agent tools.

    This node concatenates two lists of LLM agent tools.

    An agent can be provided with a list of tools to choose from. Use this
    node to concatenate existing tools into a list, which can then be provided to an agent.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the embeddings connector node(s) for tools,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        spec_one: ToolListPortObjectSpec,
        spec_two: ToolListPortObjectSpec,
    ):
        spec_one.validate_context(ctx)
        spec_two.validate_context(ctx)
        return self._create_spec(spec_one, spec_two)

    def _create_spec(
        self, spec_one: ToolListPortObjectSpec, spec_two: ToolListPortObjectSpec
    ):
        return ToolListPortObjectSpec(spec_one.tool_specs + spec_two.tool_specs)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        object_one: ToolListPortObject,
        object_two: ToolListPortObject,
    ):
        tools = object_one.tools + object_two.tools
        return ToolListPortObject(
            self._create_spec(object_one.spec, object_two.spec), tools
        )
