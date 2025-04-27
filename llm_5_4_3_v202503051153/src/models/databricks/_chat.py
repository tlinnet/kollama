from typing import Dict

from knime.extension import CredentialPortObjectSpec
from ..base import (
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    GeneralRemoteSettings,
)
import knime.extension as knext
from ._utils import (
    check_workspace_available,
    get_api_key,
    get_model_choices_provider,
    get_models,
    get_workspace_port_type,
    databricks_category,
    get_base_url,
)

databricks_workspace_port_type = get_workspace_port_type()


class DatabricksChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(
        self,
        databricks_workspace_spec: knext.CredentialPortObjectSpec,
        endpoint: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n_requests: int,
    ):
        self._databricks_workspace_spec = databricks_workspace_spec
        self._endpoint = endpoint
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n_requests = n_requests

    @property
    def databricks_workspace_spec(self) -> CredentialPortObjectSpec:
        return self._databricks_workspace_spec

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def n_requests(self) -> int:
        return self._n_requests

    def validate_context(self, ctx):
        check_workspace_available(self._databricks_workspace_spec)

    def serialize(self):
        return {
            "databricks_workspace_spec": self._databricks_workspace_spec.serialize(),
            "endpoint": self._endpoint,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n_requests": self.n_requests,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        databricks_workspace_spec = (
            databricks_workspace_port_type.spec_class.deserialize(
                data["databricks_workspace_spec"]
            )
        )
        return cls(
            databricks_workspace_spec=databricks_workspace_spec,
            endpoint=data["endpoint"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            max_tokens=data["max_tokens"],
            n_requests=data["n_requests"],
        )


class DatabricksChatModelPortObject(ChatModelPortObject):
    def __init__(self, spec: DatabricksChatModelPortObjectSpec):
        super().__init__(spec=spec)

    @property
    def spec(self) -> DatabricksChatModelPortObjectSpec:
        return self._spec

    def create_model(self, ctx):
        from langchain_openai import ChatOpenAI

        base_url = get_base_url(self.spec.databricks_workspace_spec)

        api_key = get_api_key(self.spec.databricks_workspace_spec)

        return ChatOpenAI(
            model=self.spec.endpoint,
            base_url=base_url,
            api_key=api_key,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
        )


databricks_chat_model_port_type = knext.port_type(
    "Databricks Chat Model",
    DatabricksChatModelPortObject,
    DatabricksChatModelPortObjectSpec,
)


class DatabricksChatModelSettings(GeneralRemoteSettings):
    max_tokens = knext.IntParameter(
        label="Maximum response length (token)",
        description="""
        The maximum number of tokens to generate.

        This value, plus the token count of your prompt, cannot exceed the model's context length.
        """,
        default_value=200,
        min_value=1,
    )


@knext.node(
    "Databricks Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/Databricks-chat-model-connector.png",
    category=databricks_category,
    keywords=["Databricks", "GenAI", "Chat model"],
)
@knext.input_port(
    "Databricks Workspace",
    "Credentials for a Databricks workspace.",
    databricks_workspace_port_type,
)
@knext.output_port(
    "Databricks Chat Model",
    "Connection to a chat model served by a Databricks workspace.",
    databricks_chat_model_port_type,
)
class DatabricksChatModelConnector:
    """Connects to chat models served by a Databricks workspace.

    This node connects to a chat model served by the Databricks workspace that is provided as an input.
    See the
    [Databricks documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html)
    for more information on how to serve a model in a Databricks workspace.

    **Note**: This node is only available if the
    [KNIME Databricks Integration](https://hub.knime.com/knime/extensions/org.knime.features.bigdata.databricks/latest)
    is installed.
    """

    # TODO investigate if it's possible to list the available models
    endpoint = knext.StringParameter(
        "Endpoint",
        "The name of the endpoint of the model in the Databricks workspace.",
        default_value="",
        choices=get_model_choices_provider("chat"),
    )

    model_settings = DatabricksChatModelSettings()

    def configure(
        self, ctx, databricks_workspace_spec
    ) -> DatabricksChatModelPortObjectSpec:
        if self.endpoint == "":
            raise knext.InvalidParametersError("Select a chat model endpoint.")
        check_workspace_available(databricks_workspace_spec)
        return self._create_output_spec(databricks_workspace_spec)

    def _create_output_spec(
        self, databricks_workspace_spec
    ) -> DatabricksChatModelPortObjectSpec:
        return DatabricksChatModelPortObjectSpec(
            databricks_workspace_spec=databricks_workspace_spec,
            endpoint=self.endpoint,
            temperature=self.model_settings.temperature,
            top_p=self.model_settings.top_p,
            max_tokens=self.model_settings.max_tokens,
            n_requests=self.model_settings.n_requests,
        )

    def execute(self, ctx, databricks_workspace) -> DatabricksChatModelPortObject:
        if self.endpoint not in get_models(databricks_workspace.spec, "chat"):
            raise knext.InvalidParametersError(
                f"The chat model '{self.endpoint}' is not served by the Databricks workspace."
            )
        return DatabricksChatModelPortObject(
            self._create_output_spec(databricks_workspace.spec)
        )
