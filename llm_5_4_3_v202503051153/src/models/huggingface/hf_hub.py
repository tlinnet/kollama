# KNIME / own imports
import knime.extension as knext
from ..base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObject,
    ChatModelPortObjectSpec,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    CredentialsSettings,
    AIPortObjectSpec,
)
from .hf_base import (
    hf_category,
    hf_icon,
    HFPromptTemplateSettings,
    HFModelSettings,
    raise_for,
)


hf_hub_category = knext.category(
    path=hf_category,
    level_id="hub",
    name="Hub",
    description="Contains nodes that connect to Hugging Face Hub.",
    icon=hf_icon,
)


def _create_repo_id_parameter() -> knext.StringParameter:
    return knext.StringParameter(
        label="Repo ID",
        description="""The model name to be used, in the format `<organization_name>/<model_name>`. For example, 
        `mistralai/Mistral-7B-Instruct-v0.3` for text generation, or `sentence-transformers/all-MiniLM-L6-v2`
        for embedding model.

        You can find available models at the [Hugging Face Models repository](https://huggingface.co/models).""",
        default_value="",
    )


@knext.parameter_group(label="Hugging Face Hub Settings")
class HFHubSettings:
    repo_id = _create_repo_id_parameter()


class HFAuthenticationPortObjectSpec(AIPortObjectSpec):
    def __init__(self, credentials: str) -> None:
        super().__init__()
        self._credentials = credentials

    @property
    def credentials(self) -> str:
        return self._credentials

    def get_token(self, ctx: knext.ExecutionContext):
        return ctx.get_credentials(self.credentials).password

    def validate_context(self, ctx: knext.ConfigurationContext):
        if self.credentials not in ctx.get_credential_names():
            raise knext.InvalidParametersError(
                f"""The selected credentials '{self.credentials}' holding the Hugging Face Hub API key do not exist. 
                Make sure that you have selected the correct credentials and that they are still available."""
            )
        hub_token = ctx.get_credentials(self.credentials)
        if not hub_token.password:
            raise knext.InvalidParametersError(
                f"""The Hugging Face Hub API key '{self.credentials}' does not exist. Make sure that the node you are using to pass the credentials 
                (e.g. the Credentials Configuration node) is still passing the valid API key as a flow variable to the downstream nodes."""
            )

    def serialize(self) -> dict:
        return {"credentials": self._credentials}

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["credentials"])


class HFAuthenticationPortObject(knext.PortObject):
    def __init__(self, spec: HFAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize(cls, spec: HFAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


class HFHubLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        credentials: HFAuthenticationPortObjectSpec,
        repo_id,
        n_requests,
        model_kwargs,
    ) -> None:
        super().__init__()
        self._credentials = credentials
        self._repo_id = repo_id
        self._n_requests = n_requests
        self._model_kwargs = model_kwargs

    def validate_context(self, ctx: knext.ConfigurationContext):
        return self._credentials.validate_context(ctx)

    @property
    def credentials(self) -> HFAuthenticationPortObjectSpec:
        return self._credentials

    @property
    def repo_id(self):
        return self._repo_id

    @property
    def n_requests(self):
        return self._n_requests

    @property
    def model_kwargs(self):
        return self._model_kwargs

    def serialize(self) -> dict:
        return {
            **self._credentials.serialize(),
            "repo_id": self._repo_id,
            "n_requests": self._n_requests,
            "model_kwargs": self._model_kwargs,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            HFAuthenticationPortObjectSpec.deserialize(data),
            data["repo_id"],
            data.get("n_requests", 1),
            data["model_kwargs"],
        )


class HFHubLLMPortObject(LLMPortObject):
    def __init__(self, spec: HFHubLLMPortObjectSpec) -> None:
        super().__init__(spec)

    @property
    def spec(self) -> HFHubLLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from ._hf_llm import HFLLM

        return HFLLM(
            model=self.spec.repo_id,
            hf_api_token=self.spec.credentials.get_token(ctx),
            max_new_tokens=self.spec.model_kwargs.get("max_new_tokens"),
            top_k=self.spec.model_kwargs.get("top_k"),
            top_p=self.spec.model_kwargs.get("top_p"),
            typical_p=self.spec.model_kwargs.get("typical_p"),
            repetition_penalty=self.spec.model_kwargs.get("repetition_penalty"),
            temperature=self.spec.model_kwargs.get("temperature"),
        )


class HFHubChatModelPortObjectSpec(HFHubLLMPortObjectSpec, ChatModelPortObjectSpec):
    def __init__(
        self,
        llm_spec: HFHubLLMPortObjectSpec,
        system_prompt_template: str,
        prompt_template: str,
    ) -> None:
        super().__init__(
            llm_spec._credentials,
            llm_spec.repo_id,
            llm_spec.n_requests,
            llm_spec.model_kwargs,
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
        data["system_prompt_template"] = self.system_prompt_template
        data["prompt_template"] = self.prompt_template
        return data

    @classmethod
    def deserialize(cls, data: dict):
        llm_spec = HFHubLLMPortObjectSpec.deserialize(data)
        return cls(
            llm_spec,
            system_prompt_template=data["system_prompt_template"],
            prompt_template=data["prompt_template"],
        )


class HFHubChatModelPortObject(HFHubLLMPortObject, ChatModelPortObject):
    @property
    def spec(self) -> HFHubChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from .._adapter import LLMChatModelAdapter

        llm = super().create_model(ctx)
        return LLMChatModelAdapter(
            llm=llm,
            system_prompt_template=self.spec.system_prompt_template,
            prompt_template=self.spec.prompt_template,
        )


class HFHubEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self, hub_credentials: HFAuthenticationPortObjectSpec, repo_id: str
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._hub_credentials = hub_credentials

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def hub_credentials_name(self) -> str:
        return self._hub_credentials.credentials

    def validate_context(self, ctx: knext.ConfigurationContext):
        self._hub_credentials.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            **self._hub_credentials.serialize(),
            "repo_id": self.repo_id,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(HFAuthenticationPortObjectSpec.deserialize(data), data["repo_id"])


class HFHubEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> HFHubEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from langchain_community.embeddings import HuggingFaceHubEmbeddings

        hub_api_token = ctx.get_credentials(self.spec.hub_credentials_name).password
        return HuggingFaceHubEmbeddings(
            repo_id=self.spec.repo_id, huggingfacehub_api_token=hub_api_token
        )


hf_authentication_port_type = knext.port_type(
    "HF Hub Authentication",
    HFAuthenticationPortObject,
    HFAuthenticationPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HuggingFaceAuthenticationPortObject",
)


@knext.node(
    "HF Hub Authenticator",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_hub_category,
    id="HuggingFaceHubAuthenticator",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
    ],
)
@knext.output_port(
    "HF Hub Authentication",
    "Validated authentication for Hugging Face Hub.",
    hf_authentication_port_type,
)
class HFHubAuthenticator:
    """
    Authenticates the Hugging Face Hub API key.

    This node provides the authentication for all Hugging Face Hub models.

    It allows you to select the credentials that contain a valid Hugging Face Hub API key in their *password* field (the *username* is ignored).
    Credentials can be set on the workflow level (right-click the workflow in the KNIME Explorer and click "Workflow Credentials")
    or created inside the workflow e.g. with the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory) and fed into this node via a flow variable.

    If you do not have a Hugging Face API key yet, you can generate one by visiting [Hugging Face](https://huggingface.co/settings/tokens).
    Follow the instructions provided to generate your API key.
    """

    credentials_settings = CredentialsSettings(
        label="Hugging Face API key",
        description="""
            The credentials containing the Hugging Face Hub API key in its *password* field (the *username* is ignored).
            """,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> HFAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            # credentials_settings.credentials_param = "Placeholder value"
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(self, ctx: knext.ExecutionContext) -> HFAuthenticationPortObject:
        # import from _session to ensure that it is configured with the timeout
        from ._session import huggingface_hub

        try:
            huggingface_hub.whoami(
                ctx.get_credentials(
                    self.credentials_settings.credentials_param
                ).password
            )
        except Exception as e:
            raise_for(e, knext.InvalidParametersError("Invalid API Key."))

        return HFAuthenticationPortObject(self.create_spec())

    def create_spec(self) -> HFAuthenticationPortObjectSpec:
        return HFAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
        )


hf_hub_llm_port_type = knext.port_type(
    "HF LLM",
    HFHubLLMPortObject,
    HFHubLLMPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HuggingFaceHubLLMPortObject",
)


@knext.node(
    "HF Hub LLM Connector",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_hub_category,
    id="HuggingFaceHubConnector",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
        "Large Language Model",
    ],
)
@knext.input_port(
    "HF Authentication",
    "Validated authentication for Hugging Face Hub.",
    hf_authentication_port_type,
)
@knext.output_port(
    "HF LLM",
    "Connection to a specific LLM from Hugging Face Hub.",
    hf_hub_llm_port_type,
)
class HFHubConnector:
    """
    Connects to an LLM hosted on the Hugging Face Hub.

    This node establishes a connection to a specific LLM hosted on the Hugging Face Hub.
    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator** node.

    Provide the name of the desired LLM repository available on the [Hugging Face Hub](https://huggingface.co/models) as an input.

    For more details and information about integrating LLMs from the Hugging Face Hub, refer to the
    [LangChain documentation](https://python.langchain.com/docs/modules/model_io/models/llms/integrations/huggingface_hub).

    Please ensure that you have the necessary permissions to access the model.
    Failures with gated models may occur due to outdated tokens.

    **Note**: If you use the [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    hub_settings = HFHubSettings()
    model_settings = HFModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        huggingface_auth_spec: HFAuthenticationPortObjectSpec,
    ) -> HFHubLLMPortObjectSpec:
        if not self.hub_settings.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        huggingface_auth_spec.validate_context(ctx)
        _validate_repo_id(
            self.hub_settings.repo_id, huggingface_auth_spec.get_token(ctx)
        )
        return self.create_spec(huggingface_auth_spec)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        huggingface_auth: HFAuthenticationPortObject,
    ) -> HFHubLLMPortObject:
        return HFHubLLMPortObject(self.create_spec(huggingface_auth.spec))

    def create_spec(
        self, huggingface_auth_spec: HFAuthenticationPortObjectSpec
    ) -> HFHubLLMPortObjectSpec:
        model_kwargs = {
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "top_k": self.model_settings.top_k,
            "typical_p": self.model_settings.typical_p,
            "repetition_penalty": self.model_settings.repetition_penalty,
            "max_new_tokens": self.model_settings.max_new_tokens,
        }

        return HFHubLLMPortObjectSpec(
            huggingface_auth_spec,
            self.hub_settings.repo_id,
            self.model_settings.n_requests,
            model_kwargs,
        )


hf_hub_chat_model_port_type = knext.port_type(
    "HF Hub Chat Model",
    HFHubChatModelPortObject,
    HFHubChatModelPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HFHubChatModelPortObject",
)


@knext.node(
    "HF Hub Chat Model Connector",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_hub_category,
    id="HFHubChatModelConnector",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
    ],
)
@knext.input_port(
    "HF Authentication",
    "Validated authentication for Hugging Face Hub.",
    hf_authentication_port_type,
)
@knext.output_port(
    "HF Hub Chat Model",
    "Connection to a specific chat model from Hugging Face Hub.",
    hf_hub_chat_model_port_type,
)
class HFHubChatModelConnector:
    """
    Connects to a chat model hosted on the Hugging Face Hub.

    This node establishes a connection to a specific chat model hosted on the Hugging Face Hub.
    The difference to the HF Hub LLM Connector is that this node allows you to provide prompt templates which are crucial for
    obtaining the best output from many models that have been fine-tuned for chat-based usecases.

    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator** node.

    Provide the name of the desired chat model repository available on the
    [Hugging Face Hub](https://huggingface.co/models)
    as an input.

    Please ensure that you have the necessary permissions to access the model.
    Failures with gated models may occur due to outdated tokens.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    hub_settings = HFHubSettings()
    template_settings = HFPromptTemplateSettings()
    model_settings = HFModelSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        auth: HFAuthenticationPortObjectSpec,
    ) -> HFHubChatModelPortObjectSpec:
        auth.validate_context(ctx)
        if not self.hub_settings.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        _validate_repo_id(self.hub_settings.repo_id, auth.get_token(ctx))
        return self._create_spec(auth)

    def _create_spec(
        self, auth: HFAuthenticationPortObjectSpec
    ) -> HFHubChatModelPortObjectSpec:
        model_kwargs = {
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "top_k": self.model_settings.top_k,
            "typical_p": self.model_settings.typical_p,
            "repetition_penalty": self.model_settings.repetition_penalty,
            "max_new_tokens": self.model_settings.max_new_tokens,
        }

        llm_spec = HFHubLLMPortObjectSpec(
            auth,
            self.hub_settings.repo_id,
            self.model_settings.n_requests,
            model_kwargs,
        )
        return HFHubChatModelPortObjectSpec(
            llm_spec,
            self.template_settings.system_prompt_template,
            self.template_settings.prompt_template,
        )

    def execute(
        self, ctx, auth: HFAuthenticationPortObject
    ) -> HFHubChatModelPortObject:
        return HFHubChatModelPortObject(self._create_spec(auth.spec))


def _validate_repo_id(repo_id: str, token: str):
    # import from _session to ensure that it is configured with the timeout
    from ._session import huggingface_hub

    try:
        huggingface_hub.model_info(repo_id, token=token)
    except huggingface_hub.utils._errors.GatedRepoError as e:
        raise knext.InvalidParametersError(
            f"""Access to this repository is restricted: {e}. 
            Please make sure you have the necessary permissions to access the model and an up-to-date token."""
        )
    except Exception as e:
        raise_for(
            e, knext.InvalidParametersError(f"Please provide a valid repo ID. {e}")
        )


hf_embeddings_port_type = knext.port_type(
    "HF Hub Embeddings",
    HFHubEmbeddingsPortObject,
    HFHubEmbeddingsPortObjectSpec,
    id="org.knime.python.llm.models.huggingface.HFHubEmbeddingsPortObject",
)


@knext.node(
    "HF Hub Embeddings Connector",
    knext.NodeType.SOURCE,
    hf_icon,
    category=hf_hub_category,
    id="HFHubEmbeddingsConnector",
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "HuggingFace",
        "Hugging Face",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "HF Hub Authentication",
    "The authentication for the Hugging Face Hub.",
    hf_authentication_port_type,
)
@knext.output_port(
    "HF Hub Embeddings",
    "An embeddings model connected to Hugging Face Hub.",
    hf_embeddings_port_type,
)
class HFHubEmbeddingsConnector:
    """
    Connects to an embedding model hosted on the Hugging Face Hub.

    This node establishes a connection to a specific embedding model hosted on the Hugging Face Hub.

    To use this node, you need to successfully authenticate with the Hugging Face Hub using the **HF Hub Authenticator** node.

    Provide the name of the desired embeddings repository available on the [Hugging Face Hub](https://huggingface.co/) as an input.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    repo_id = _create_repo_id_parameter()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        authentication_spec: HFAuthenticationPortObjectSpec,
    ) -> HFHubEmbeddingsPortObjectSpec:
        if not self.repo_id:
            raise knext.InvalidParametersError("Please enter a repo ID.")
        authentication_spec.validate_context(ctx)
        _validate_repo_id(self.repo_id, authentication_spec.get_token(ctx))
        return self.create_spec(authentication_spec)

    def create_spec(
        self, authentication_spec: HFAuthenticationPortObjectSpec
    ) -> HFHubEmbeddingsPortObjectSpec:
        return HFHubEmbeddingsPortObjectSpec(authentication_spec, self.repo_id)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        authentication: HFAuthenticationPortObject,
    ) -> HFHubEmbeddingsPortObject:
        output = HFHubEmbeddingsPortObject(self.create_spec(authentication.spec))
        # TODO validate that repo does supports what we want to do
        output.create_model(ctx)
        return output
