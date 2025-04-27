# KNIME / own imports
import knime.extension as knext
from .base import model_category, CredentialsSettings

from models.openai import (
    OpenAIGeneralSettings,
    OpenAIAuthenticationPortObjectSpec,
    OpenAIAuthenticationPortObject,
    OpenAIModelPortObjectSpec,
    OpenAILLMPortObjectSpec,
    OpenAILLMPortObject,
    OpenAIChatModelPortObjectSpec,
    OpenAIChatModelPortObject,
    OpenAIEmbeddingsPortObjectSpec,
    OpenAIEmbeddingsPortObject,
)

# Other imports
from socket import gaierror
import logging


azure_icon = "icons/azure_logo.png"
azure_openai_category = knext.category(
    path=model_category,
    level_id="azure",
    name="Azure OpenAI",
    description="Contains nodes for connecting to Azure OpenAI.",
    icon=azure_icon,
)

LOGGER = logging.getLogger(__name__)

# == Port Objects ==


class AzureOpenAIAuthenticationPortObjectSpec(OpenAIAuthenticationPortObjectSpec):
    def __init__(
        self,
        credentials: str,
        base_url: str,
        api_version: str,
        api_type: str,
    ) -> None:
        super().__init__(credentials, base_url)
        self._api_version = api_version
        self._api_type = api_type

    @property
    def api_version(self) -> str:
        return self._api_version

    @property
    def api_type(self) -> str:
        return self._api_type

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "api_version": self._api_version,
            "api_type": self._api_type,
        }

    @classmethod
    def deserialize(cls, data: dict):
        base_url = data["base_url"] if "base_url" in data else data["api_base"]
        return cls(data["credentials"], base_url, data["api_version"], data["api_type"])


class AzureOpenAIAuthenticationPortObject(OpenAIAuthenticationPortObject):
    def __init__(self, spec: AzureOpenAIAuthenticationPortObjectSpec) -> None:
        super().__init__(spec)

    @classmethod
    def deserialize(cls, spec: AzureOpenAIAuthenticationPortObjectSpec, storage: bytes):
        return cls(spec)


azure_openai_authentication_port_type = knext.port_type(
    "Azure OpenAI Authentication",
    AzureOpenAIAuthenticationPortObject,
    AzureOpenAIAuthenticationPortObjectSpec,
)


class AzureOpenAIModelPortObjectSpec(OpenAIModelPortObjectSpec):
    def __init__(
        self, azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec
    ) -> None:
        self._credentials = azure_auth_spec

    @property
    def api_version(self) -> str:
        return self._credentials._api_version

    @property
    def api_type(self) -> str:
        return self._credentials._api_type

    @classmethod
    def deserialize_credentials_spec(
        cls, data: dict
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        return AzureOpenAIAuthenticationPortObjectSpec.deserialize(data)


class AzureOpenAILLMPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAILLMPortObjectSpec
):
    def __init__(
        self,
        credentials: AzureOpenAIAuthenticationPortObjectSpec,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        n_requests: int,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._n_requests = n_requests

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
            data["temperature"],
            data["top_p"],
            data["max_tokens"],
            data.get("seed", 0),
            data.get("n_requests", 1),
        )


class AzureOpenAILLMPortObject(OpenAILLMPortObject):
    def __init__(self, spec: AzureOpenAILLMPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> AzureOpenAILLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from langchain_openai import AzureOpenAI

        return AzureOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            api_version=self.spec.api_version,
            azure_endpoint=self.spec.base_url,
            openai_api_type=self.spec.api_type,
            deployment_name=self.spec.model,
            temperature=self.spec.temperature,
            top_p=self.spec.top_p,
            max_tokens=self.spec.max_tokens,
            seed=self.spec.seed,
        )


azure_openai_llm_port_type = knext.port_type(
    "Azure OpenAI LLM", AzureOpenAILLMPortObject, AzureOpenAILLMPortObjectSpec
)


class AzureOpenAIChatModelPortObjectSpec(
    AzureOpenAILLMPortObjectSpec, OpenAIChatModelPortObjectSpec
):
    """Spec of an Azure OpenAI chat model."""


class AzureOpenAIChatModelPortObject(OpenAIChatModelPortObject):
    @property
    def spec(self) -> AzureOpenAIChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from langchain_openai import AzureChatOpenAI

        model_kwargs = {"top_p": self.spec.top_p}

        return AzureChatOpenAI(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            openai_api_version=self.spec.api_version,
            azure_endpoint=self.spec.base_url,
            openai_api_type=self.spec.api_type,
            deployment_name=self.spec.model,
            temperature=self.spec.temperature,
            model_kwargs=model_kwargs,
            max_tokens=self.spec.max_tokens,
            seed=self.spec.seed,
        )


azure_openai_chat_port_type = knext.port_type(
    "Azure OpenAI Chat Model",
    AzureOpenAIChatModelPortObject,
    AzureOpenAIChatModelPortObjectSpec,
)


class AzureOpenAIEmbeddingsPortObjectSpec(
    AzureOpenAIModelPortObjectSpec, OpenAIEmbeddingsPortObjectSpec
):
    def __init__(
        self,
        credentials: AzureOpenAIAuthenticationPortObjectSpec,
        model_name,
    ) -> None:
        super().__init__(credentials)
        self._model = model_name

    def serialize(self) -> dict:
        return {
            **self._credentials.serialize(),
            "model": self._model,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            AzureOpenAIAuthenticationPortObjectSpec.deserialize(data),
            data["model"],
        )


class AzureOpenAIEmbeddingsPortObject(OpenAIEmbeddingsPortObject):
    def __init__(self, spec: AzureOpenAIEmbeddingsPortObjectSpec):
        super().__init__(spec)

    @property
    def spec(self) -> AzureOpenAIEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, ctx: knext.ExecutionContext):
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            openai_api_key=ctx.get_credentials(self.spec.credentials).password,
            azure_endpoint=self.spec.base_url,
            api_version=self.spec.api_version,
            openai_api_type=self.spec.api_type,
            deployment=self.spec.model,
            chunk_size=16,  # Azure only supports 16 docs per request
        )


azure_openai_embeddings_port_type = knext.port_type(
    "Azure OpenAI Embeddings Model",
    AzureOpenAIEmbeddingsPortObject,
    AzureOpenAIEmbeddingsPortObjectSpec,
)


@knext.parameter_group(label="Azure Connection")
class AzureSettings:
    api_base = knext.StringParameter(
        label="Azure Resource Endpoint",
        description="""The Azure Resource Endpoint address can be found in the 'Keys and Endpoints' 
        section of the [Azure Portal](https://portal.azure.com/).""",
        default_value="",
    )

    api_version = knext.StringParameter(
        label="Azure API Version",
        description="""Available API versions can be found at the [Azure OpenAI API preview lifecycle](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation).
        Note that the latest API versions could support more functionality, such as function calling.""",
        default_value="2023-07-01-preview",
    )


@knext.parameter_group(label="Azure Deployment")
class AzureDeploymentSettings:
    deployment_name = knext.StringParameter(
        label="Deployment name",
        description="""The name of the deployed model to use. Find the deployed models on the [Azure AI Studio](https://oai.azure.com).""",
        default_value="",
    )


# == Nodes ==


@knext.node(
    "Azure OpenAI Authenticator",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI"],
)
@knext.output_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
class AzureOpenAIAuthenticator:
    """
    Authenticates the Azure OpenAI API key against the the Cognitive Services account.

    This node provides the authentication for all Azure OpenAI models.
    It allows you to select the credentials that contain a valid Azure OpenAI API key in their *password* field (the *username* is ignored).

    Credentials can be set on the workflow level (right-click the workflow in the KNIME Explorer and click "Workflow Credentials") or created inside the workflow e.g. with the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and fed into this node via flow variable.

    To find your Azure OpenAI API key, navigate to your Azure OpenAI Resource on the [Azure Portal](https://portal.azure.com/) and copy one of the keys from
    'Resource Management - Keys and Endpoints'.

    The Azure Resource Endpoint URL can also be found on the [Azure Portal](https://portal.azure.com/) under 'Resource Management - Keys and Endpoints'.

    For the correct API version, refer to the [Azure OpenAI API preview lifecycle](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation).
    """

    credentials_settings = CredentialsSettings(
        label="Azure OpenAI API Key",
        description="""
        The credentials containing the OpenAI API key in its *password* field (the *username* is ignored).
        """,
    )

    azure_connection = AzureSettings()

    verify_settings = knext.BoolParameter(
        "Verify settings",
        "Whether to verify the settings by calling the *list models* endpoint.",
        True,
        since_version="5.2.1",
        is_advanced=True,
    )

    def configure(
        self, ctx: knext.ConfigurationContext
    ) -> AzureOpenAIAuthenticationPortObjectSpec:
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("Credentials not provided.")

        if not self.credentials_settings.credentials_param:
            raise knext.InvalidParametersError("Credentials not selected.")

        if not self.azure_connection.api_base:
            raise knext.InvalidParametersError("API endpoint not provided.")

        if not self.azure_connection.api_version:
            raise knext.InvalidParametersError("API version not provided.")

        spec = self.create_spec()
        spec.validate_context(ctx)
        return spec

    def execute(
        self, ctx: knext.ExecutionContext
    ) -> AzureOpenAIAuthenticationPortObject:
        if self.verify_settings:
            self._verify_settings(ctx)

        return AzureOpenAIAuthenticationPortObject(self.create_spec())

    def _verify_settings(self, ctx):
        import openai

        try:
            openai.AzureOpenAI(
                api_key=ctx.get_credentials(
                    self.credentials_settings.credentials_param,
                ).password,
                api_version=self.azure_connection.api_version,
                azure_endpoint=self.azure_connection.api_base,
            ).models.list()

        except openai.APIConnectionError:
            raise knext.InvalidParametersError(
                f"Invalid Azure endpoint provided: '{self.azure_connection.api_base}'"
            )
        except openai.NotFoundError:
            raise knext.InvalidParametersError(
                """API resource not found. Please ensure you are using a valid address and API version.
                The address can be found in the 'Keys and Endpoints' section of the Azure Portal at https://portal.azure.com/.
                For the correct API version, refer to the API Version Deprecation Guide at https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation.
                """
            )
        except openai.AuthenticationError:
            raise knext.InvalidParametersError("Invalid API key provided.")

    def create_spec(self) -> AzureOpenAIAuthenticationPortObjectSpec:
        return AzureOpenAIAuthenticationPortObjectSpec(
            self.credentials_settings.credentials_param,
            self.azure_connection.api_base,
            self.azure_connection.api_version,
            "azure",
        )


@knext.node(
    "Azure OpenAI LLM Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
        "OpenAI",
        "Azure",
    ],
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI LLM",
    "Configured Azure OpenAI LLM connection.",
    azure_openai_llm_port_type,
)
class AzureOpenAILLMConnector:
    """
    Connects to an Azure OpenAI Large Language Model.

    This node establishes a connection with an Azure OpenAI Large Language Model (LLM).
    After successfully authenticating using the **Azure OpenAI Authenticator** node, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'. Note that only models compatible with Azure OpenAI's Completions API will work with this node.

    **Note**: See the **Azure OpenAI Chat Model Connector** node for LLMs optimized for chat-specific usecases.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureDeploymentSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAILLMPortObjectSpec:
        if not hasattr(azure_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No deployment name provided")

        azure_auth_spec.validate_context(ctx)

        return self.create_spec(azure_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAILLMPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # AzureOpenAI() does not verify the deployment_name unless it is prompted
        # Could be done with an AD Bearer token

        return AzureOpenAILLMPortObject(
            self.create_spec(azure_auth_port.spec, self.deployment.deployment_name)
        )

    def create_spec(
        self,
        azure_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAILLMPortObjectSpec:
        return AzureOpenAILLMPortObjectSpec(
            azure_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.seed,
            self.model_settings.n_requests,
        )


@knext.node(
    "Azure OpenAI Chat Model Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=["GenAI", "Gen AI", "Generative AI", "OpenAI", "Azure"],
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Chat Model",
    "Configured Azure OpenAI Chat Model connection.",
    azure_openai_chat_port_type,
)
class AzureOpenAIChatModelConnector:
    """
    Connects to an Azure OpenAI Chat Model.

    This node establishes a connection with an Azure OpenAI Chat Model.
    After successfully authenticating using the **Azure OpenAI Authenticator** node, enter the deployment name of
    the model you want to use. You can find the models on the [Azure AI Studio](https://oai.azure.com) at
    'Management - Deployments'.

    **Note**: Chat models are LLMs that have been fine-tuned for chat-based usecases. As such, these models can also be
    used in other applications as well.

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureDeploymentSettings()
    model_settings = OpenAIGeneralSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAIChatModelPortObjectSpec:
        if not hasattr(azure_openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No model name provided.")

        azure_openai_auth_spec.validate_context(ctx)

        return self.create_spec(azure_openai_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_openai_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAIChatModelPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # AzureChatOpenAI() does not verify the deployment_name until it is prompted

        return AzureOpenAIChatModelPortObject(
            self.create_spec(
                azure_openai_auth_port.spec, self.deployment.deployment_name
            )
        )

    def create_spec(
        self,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAIChatModelPortObjectSpec:
        return AzureOpenAIChatModelPortObjectSpec(
            azure_openai_auth_spec,
            deployment_name,
            self.model_settings.temperature,
            self.model_settings.top_p,
            self.model_settings.max_tokens,
            self.model_settings.seed,
            self.model_settings.n_requests,
        )


@knext.node(
    "Azure OpenAI Embeddings Connector",
    knext.NodeType.SOURCE,
    azure_icon,
    category=azure_openai_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "OpenAI",
        "Azure",
        "RAG",
        "Retrieval Assisted Generation",
    ],
)
@knext.input_port(
    "Azure OpenAI Authentication",
    "Validated authentication for Azure OpenAI.",
    azure_openai_authentication_port_type,
)
@knext.output_port(
    "Azure OpenAI Embeddings Model",
    "Configured Azure OpenAI Embeddings Model connection.",
    azure_openai_embeddings_port_type,
)
class AzureOpenAIEmbeddingsConnector:
    """
    Connects to an Azure OpenAI Embeddings Model.

    This node establishes a connection with an Azure OpenAI Embeddings Model. After successfully authenticating
    using the **Azure OpenAI Authenticator** node, you need to provide the name of a deployed embeddings model
    found on the [Azure AI Studio](https://oai.azure.com).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials
    flow variable was not saved and will therefore not be available to downstream nodes.
    """

    deployment = AzureDeploymentSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
    ) -> AzureOpenAIEmbeddingsPortObjectSpec:
        if not hasattr(azure_openai_auth_spec, "api_type"):
            raise knext.InvalidParametersError("Use OpenAI Model Connectors")

        if not self.deployment.deployment_name:
            raise knext.InvalidParametersError("No deployment name provided")

        azure_openai_auth_spec.validate_context(ctx)

        return self.create_spec(azure_openai_auth_spec, self.deployment.deployment_name)

    def execute(
        self,
        ctx: knext.ExecutionContext,
        azure_openai_auth_port: AzureOpenAIAuthenticationPortObject,
    ) -> AzureOpenAIEmbeddingsPortObject:
        # We know that the API key is correct from the authenticator, but a call to
        # OpenAIEmeddings() does not verify the deployment_name unless it is prompted

        return AzureOpenAIEmbeddingsPortObject(
            self.create_spec(
                azure_openai_auth_port.spec, self.deployment.deployment_name
            )
        )

    def create_spec(
        self,
        azure_openai_auth_spec: AzureOpenAIAuthenticationPortObjectSpec,
        deployment_name: str,
    ) -> AzureOpenAIEmbeddingsPortObjectSpec:
        return AzureOpenAIEmbeddingsPortObjectSpec(
            azure_openai_auth_spec, deployment_name
        )
