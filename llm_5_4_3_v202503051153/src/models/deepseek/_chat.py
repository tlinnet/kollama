import knime.extension as knext
from ..base import ChatModelPortObject, ChatModelPortObjectSpec, OutputFormatOptions
from ._base import deepseek_icon, deepseek_category
from ._auth import (
    DeepSeekAuthenticationPortObjectSpec,
    DeepSeekAuthenticationPortObject,
    deepseek_auth_port_type,
)


class DeepSeekChatModelPortObjectSpec(ChatModelPortObjectSpec):
    """Spec of a DeepSeek Chat Model"""

    def __init__(
        self,
        auth: DeepSeekAuthenticationPortObjectSpec,
        model: str,
        temperature: float,
        max_tokens: int,
        n_requests=1,
    ):
        super().__init__(n_requests)
        self._auth = auth
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def auth(self) -> DeepSeekAuthenticationPortObjectSpec:
        return self._auth

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self._auth.serialize(),
            "n_requests": self._n_requests,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict):
        auth = DeepSeekAuthenticationPortObjectSpec.deserialize(data["auth"])
        return cls(
            auth=auth,
            model=data["model"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            n_requests=data.get("n_requests", 1),
        )


class DeepSeekChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> DeepSeekChatModelPortObjectSpec:
        return super().spec

    def create_model(
        self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text,
    ):
        from langchain_openai import ChatOpenAI

        if "reasoner" in self.spec.model:
            return ChatOpenAI(
                openai_api_key=ctx.get_credentials(self.spec.auth.credentials).password,
                base_url=self.spec.auth.base_url,
                model=self.spec.model,
                temperature=1,
                max_completion_tokens=self.spec.max_tokens,
            )

        return ChatOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.auth.credentials).password,
            base_url=self.spec.auth.base_url,
            model=self.spec.model,
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
        )


deepseek_chat_model_port_type = knext.port_type(
    "DeepSeek Chat Model", DeepSeekChatModelPortObject, DeepSeekChatModelPortObjectSpec
)


def _list_models(ctx: knext.ConfigurationContext):
    if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
        return auth_spec.get_model_list(ctx)
    return ["deepseek-chat", "deepseek-reasoner"]


@knext.node(
    name="DeepSeek Chat Model Connector",
    node_type=knext.NodeType.SOURCE,
    icon_path=deepseek_icon,
    category=deepseek_category,
    keywords=["DeepSeek", "GenAI", "Reasoning"],
)
@knext.input_port(
    "DeepSeek Authentication",
    "The authentication for the DeepSeek API.",
    deepseek_auth_port_type,
)
@knext.output_port(
    "DeepSeek Chat Model",
    "The DeepSeek chat model which can be used in the LLM Prompter and Chat Model Prompter.",
    deepseek_chat_model_port_type,
)
class DeepSeekChatModelConnector:
    """Connects to a chat model provided by the DeepSeek API.

    This node establishes a connection with a DeepSeek Chat Model. After successfully authenticating
    using the **DeepSeek Authenticator** node, you can select a chat model from a predefined list.

    **Note**: Data sent to the DeepSeek API is stored and used for training future models.
    """

    model = knext.StringParameter(
        "Model",
        description="The model to use. The available models are fetched from the DeepSeek API if possible.",
        default_value="deepseek-chat",
        choices=_list_models,
    )

    temperature = knext.DoubleParameter(
        "Temperature",
        description="""
        Sampling temperature to use, between 0.0 and 2.0.

        Higher values will lead to less deterministic but more creative answers.
        Recommended values for different tasks:

        - Coding / math: 0.0
        - Data cleaning / data analysis: 1.0
        - General conversation: 1.3
        - Translation: 1.3
        - Creative writing: 1.5
        """,
        default_value=1,
    )

    max_tokens = knext.IntParameter(
        "Max Tokens",
        description="The maximum number of tokens to generate in the response",
        default_value=4096,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: DeepSeekAuthenticationPortObjectSpec,
    ) -> DeepSeekChatModelPortObjectSpec:
        auth.validate_context(ctx)
        return self.create_spec(auth)

    def create_spec(
        self, auth: DeepSeekAuthenticationPortObjectSpec
    ) -> DeepSeekChatModelPortObjectSpec:
        return DeepSeekChatModelPortObjectSpec(
            auth=auth,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def execute(
        self, ctx: knext.ExecutionContext, auth: DeepSeekAuthenticationPortObject
    ) -> DeepSeekChatModelPortObject:
        return DeepSeekChatModelPortObject(self.create_spec(auth.spec))
