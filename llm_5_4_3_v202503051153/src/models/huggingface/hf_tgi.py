# KNIME / own imports
from typing import Optional

import knime.extension as knext
from ..base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObject,
    ChatModelPortObjectSpec,
)
from .hf_base import (
    hf_category,
    hf_icon,
    HFPromptTemplateSettings,
    HFModelSettings,
)
from .hf_hub import (
    HFAuthenticationPortObject,
    HFAuthenticationPortObjectSpec,
    hf_authentication_port_type,
)

hf_tgi_category = knext.category(
    path=hf_category,
    level_id="tgi",
    name="Text Generation Inference (TGI)",
    description="Contains nodes that connect to Hugging Face's text generation inference server.",
    icon=hf_icon,
)


@knext.parameter_group(label="Hugging Face TextGen Inference Server Settings")
class HFTGIServerSettings:
    server_url = knext.StringParameter(
        label="Inference server URL",
        description="The URL of the inference server to use, e.g. `http://localhost:8010/`.",
        default_value="",
    )


class HFTGIModelSettings(HFModelSettings):
    seed = knext.IntParameter(
        label="Seed",
        description="""
        Set the seed parameter to any integer of your choice and use the same value across requests
        to have reproducible outputs. 

        The default value of 0 means that no seed is specified.
        """,
        default_value=0,
        is_advanced=True,
        since_version="5.3.0",
    )


class HFTGILLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        inference_server_url,
        max_new_tokens,
        top_k,
        top_p,
        typical_p,
        temperature,
        repetition_penalty,
        seed,
        n_requests,
        hf_hub_auth: Optional[HFAuthenticationPortObjectSpec],
    ) -> None:
        super().__init__()
        self._inference_server_url = inference_server_url
        self._max_new_tokens = max_new_tokens
        self._top_k = top_k
        self._top_p = top_p
        self._typical_p = typical_p
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._seed = seed
        self._n_requests = n_requests
        self._hf_hub_auth = hf_hub_auth

    @property
    def inference_server_url(self):
        return self._inference_server_url

    @property
    def max_new_tokens(self):
        return self._max_new_tokens

    @property
    def top_k(self):
        return self._top_k

    @property
    def top_p(self):
        return self._top_p

    @property
    def typical_p(self):
        return self._typical_p

    @property
    def temperature(self):
        return self._temperature

    @property
    def repetition_penalty(self):
        return self._repetition_penalty

    @property
    def seed(self):
        return self._seed

    @property
    def n_requests(self):
        return self._n_requests

    @property
    def hf_hub_auth(self) -> Optional[HFAuthenticationPortObjectSpec]:
        return self._hf_hub_auth

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.hf_hub_auth:
            self.hf_hub_auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "inference_server_url": self.inference_server_url,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "n_requests": self.n_requests,
            "hf_hub_auth": self._hf_hub_auth.serialize() if self._hf_hub_auth else None,
        }

    @classmethod
    def deserialize(cls, data: dict):
        hub_auth_data = data.get("hf_hub_auth")
        if hub_auth_data:
            hub_auth = HFAuthenticationPortObjectSpec.deserialize(hub_auth_data)
        else:
            hub_auth = None
        return cls(
            data["inference_server_url"],
            data["max_new_tokens"],
            data["top_k"],
            data["top_p"],
            data["typical_p"],
            data["temperature"],
            data["repetition_penalty"],
            seed=data.get("seed", 0),
            n_requests=data.get("n_requests", 1),
            hf_hub_auth=hub_auth,
        )


class HFTGILLMPortObject(LLMPortObject):
    def __init__(self, spec: HFTGILLMPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTGILLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from ._hf_llm import HFLLM

        hub_auth = self.spec.hf_hub_auth
        return HFLLM(
            model=self.spec.inference_server_url,
            max_new_tokens=self.spec.max_new_tokens,
            top_k=self.spec.top_k,
            top_p=self.spec.top_p,
            typical_p=self.spec.typical_p,
            temperature=self.spec.temperature,
            repetition_penalty=self.spec.repetition_penalty,
            seed=self.spec.seed,
            hf_api_token=hub_auth.get_token(ctx) if hub_auth else None,
        )


hf_tgi_llm_port_type = knext.port_type(
    "Hugging Face TGI LLM",
    HFTGILLMPortObject,
    HFTGILLMPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HuggingFaceTextGenInfLLMPortObject",
)


class HFTGIChatModelPortObjectSpec(HFTGILLMPortObjectSpec, ChatModelPortObjectSpec):
    def __init__(
        self,
        llm_spec: HFTGILLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
    ) -> None:
        super().__init__(
            llm_spec.inference_server_url,
            llm_spec.max_new_tokens,
            llm_spec.top_k,
            llm_spec.top_p,
            llm_spec.typical_p,
            llm_spec.temperature,
            llm_spec.repetition_penalty,
            llm_spec.seed,
            llm_spec.n_requests,
            llm_spec.hf_hub_auth,
        )
        self._system_prompt_template = system_prompt_template
        self._prompt_template = prompt_template

    @property
    def system_prompt_template(self) -> str:
        return self._system_prompt_template

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    def serialize(self) -> dict:
        data = super().serialize()
        data["system_prompt_template"] = self._system_prompt_template
        data["prompt_template"] = self._prompt_template
        return data

    @classmethod
    def deserialize(cls, data: dict):
        llm_spec = HFTGILLMPortObjectSpec.deserialize(data)
        return cls(
            llm_spec,
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
        )


class HFTGIChatModelPortObject(HFTGILLMPortObject, ChatModelPortObject):
    def __init__(self, spec: HFTGIChatModelPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFTGIChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from .._adapter import LLMChatModelAdapter

        llm = super().create_model(ctx)
        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=self.spec.system_prompt_template,
            prompt_template=self.spec.prompt_template,
        )


huggingface_textGenInference_chat_port_type = knext.port_type(
    "Hugging Face TGI Chat Model",
    HFTGIChatModelPortObject,
    HFTGIChatModelPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HFTGIChatModelPortObject",
)


@knext.node(
    "HF TGI LLM Connector",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_tgi_category,
    id="HuggingfaceTextGenInferenceConnector",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
        "Text Generation Inference",
        "Large Language Model",
    ],
)
@knext.input_port(
    name="Hugging Face Hub Connection",
    description="An optional Hugging Face Hub connection that can be used to "
    "access protected Hugging Face inference endpoints.",
    port_type=hf_authentication_port_type,
    optional=True,
)
@knext.output_port(
    "Huggingface TGI Configuration",
    "Connection to an LLM hosted on a TGI server.",
    hf_tgi_llm_port_type,
)
class HFTGILLMConnector:
    """
    Connects to a dedicated Text Generation Inference Server.

    This node can connect to locally or remotely hosted TGI servers which includes
    [Text Generation Inference Endpoints](https://huggingface.co/docs/inference-endpoints/) of popular
    text generation models that are deployed via Hugging Face Hub.

    Protected endpoints require a connection with a **HF Hub Authenticator** node in order to authenticate with Hugging Face Hub.

    The [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
    is a Rust, Python, and gRPC server specifically designed for text generation inference.
    It can be self-hosted to power LLM APIs and inference widgets.

    For more details and information about integrating with the Hugging Face TextGen Inference
    and setting up a local server, refer to the
    [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_textgen_inference).

    **Note**: If you use the [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key via the **HF Hub Authenticator** node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    settings = HFTGIServerSettings()
    model_settings = HFTGIModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        hf_hub_auth: Optional[HFAuthenticationPortObjectSpec],
    ) -> HFTGILLMPortObjectSpec:
        if not self.settings.server_url:
            raise knext.InvalidParametersError("Server URL missing")

        if hf_hub_auth:
            hf_hub_auth.validate_context(ctx)

        return self.create_spec(hf_hub_auth)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        hf_hub_auth: Optional[HFAuthenticationPortObject],
    ) -> HFTGILLMPortObject:
        hf_hub_auth = hf_hub_auth.spec if hf_hub_auth else None
        return HFTGILLMPortObject(self.create_spec(hf_hub_auth))

    def create_spec(
        self, hf_hub_auth: Optional[HFAuthenticationPortObjectSpec]
    ) -> HFTGILLMPortObjectSpec:
        seed = None if self.model_settings.seed == 0 else self.model_settings.seed

        return HFTGILLMPortObjectSpec(
            self.settings.server_url,
            self.model_settings.max_new_tokens,
            self.model_settings.top_k,
            self.model_settings.top_p,
            self.model_settings.typical_p,
            self.model_settings.temperature,
            self.model_settings.repetition_penalty,
            seed=seed,
            n_requests=self.model_settings.n_requests,
            hf_hub_auth=hf_hub_auth,
        )


@knext.node(
    "HF TGI Chat Model Connector",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_tgi_category,
    id="HFTGIChatModelConnector",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
        "Text Generation Inference",
    ],
)
@knext.input_port(
    name="Hugging Face Hub Connection",
    description="An optional Hugging Face Hub connection that can be used to "
    "access protected Hugging Face inference endpoints.",
    port_type=hf_authentication_port_type,
    optional=True,
)
@knext.output_port(
    "Huggingface TGI LLM Connection",
    "Connection to a chat model hosted on a Text Generation Inference server.",
    huggingface_textGenInference_chat_port_type,
)
class HFTGIChatModelConnector:
    """
    Connects to a dedicated Text Generation Inference Server.

    This node can connect to locally or remotely hosted TGI servers which includes
    [Text Generation Inference Endpoints](https://huggingface.co/docs/inference-endpoints/) of popular
    text generation models that are deployed via Hugging Face Hub.

    Protected endpoints require a connection with a **HF Hub Authenticator** node in order to authenticate with Hugging Face Hub.

    The [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
    is a Rust, Python, and gRPC server specifically designed for text generation inference.
    It can be self-hosted to power LLM APIs and inference widgets.

    For more details and information about integrating with the Hugging Face TextGen Inference
    and setting up a local server, refer to the
    [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_textgen_inference).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key via the **HF Hub Authenticator** node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    settings = HFTGIServerSettings()
    templates = HFPromptTemplateSettings()
    model_settings = HFTGIModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        hf_hub_auth: Optional[HFAuthenticationPortObjectSpec],
    ) -> HFTGIChatModelPortObjectSpec:
        if not self.settings.server_url:
            raise knext.InvalidParametersError("Server URL missing")
        if hf_hub_auth:
            hf_hub_auth.validate_context(ctx)

        return self.create_spec(hf_hub_auth)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        hf_hub_auth: Optional[HFAuthenticationPortObject],
    ) -> HFTGIChatModelPortObject:
        hf_hub_auth = hf_hub_auth.spec if hf_hub_auth else None
        return HFTGIChatModelPortObject(self.create_spec(hf_hub_auth))

    def create_spec(
        self, hf_hub_auth: Optional[HFAuthenticationPortObjectSpec]
    ) -> HFTGIChatModelPortObjectSpec:
        seed = None if self.model_settings.seed == 0 else self.model_settings.seed

        llm_spec = HFTGILLMPortObjectSpec(
            self.settings.server_url,
            self.model_settings.max_new_tokens,
            self.model_settings.top_k,
            self.model_settings.top_p,
            self.model_settings.typical_p,
            self.model_settings.temperature,
            self.model_settings.repetition_penalty,
            seed=seed,
            n_requests=self.model_settings.n_requests,
            hf_hub_auth=hf_hub_auth,
        )

        return HFTGIChatModelPortObjectSpec(
            llm_spec,
            self.templates.system_prompt_template,
            self.templates.prompt_template,
        )
