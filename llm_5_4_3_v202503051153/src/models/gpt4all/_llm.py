import knime.extension as knext
from ..base import (
    LLMPortObjectSpec,
    LLMPortObject,
)

from ._base import (
    GPT4AllInputSettings,
    GPT4AllModelParameterSettings,
)

from ._utils import gpt4all_icon, gpt4all_category, is_valid_model

import logging

LOGGER = logging.getLogger(__name__)


class GPT4AllLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(
        self,
        local_path: str,
        n_threads: int,
        temperature: float,
        top_k: int,
        top_p: float,
        n_ctx: int,
        max_token: int,
        prompt_batch_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self._local_path = local_path
        self._n_threads = n_threads
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._n_ctx = n_ctx
        self._max_token = max_token
        self._prompt_batch_size = prompt_batch_size
        self._device = device

    @property
    def local_path(self) -> str:
        return self._local_path

    @property
    def n_threads(self) -> int:
        return self._n_threads

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def max_token(self) -> int:
        return self._max_token

    @property
    def prompt_batch_size(self) -> int:
        return self._prompt_batch_size

    @property
    def device(self) -> str:
        return self._device

    def serialize(self) -> dict:
        return {
            "local_path": self._local_path,
            "n_threads": self._n_threads,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "n_ctx": self._n_ctx,
            "max_token": self._max_token,
            "prompt_batch_size": self.prompt_batch_size,
            "device": self._device,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            local_path=data["local_path"],
            n_threads=data.get("n_threads", None),
            temperature=data.get("temperature", 0.2),
            top_k=data.get("top_k", 20),
            top_p=data.get("top_p", 0.15),
            n_ctx=data.get("n_ctx", 2048),
            max_token=data.get("max_token", 250),
            prompt_batch_size=data.get("prompt_batch_size", 128),
            device=data.get("device", "cpu"),
        )


class GPT4AllLLMPortObject(LLMPortObject):
    @property
    def spec(self) -> GPT4AllLLMPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        from ._gpt4all import GPT4All
        from pydantic import ValidationError

        try:
            return GPT4All(
                model=self.spec.local_path,
                n_threads=self.spec.n_threads,
                temp=self.spec.temperature,
                top_p=self.spec.top_p,
                top_k=self.spec.top_k,
                n_predict=self.spec.max_token,
                n_ctx=self.spec.n_ctx,
                n_batch=self.spec.prompt_batch_size,
                device=self.spec.device,
            )
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            if "Unable to initialize model on GPU:" in error_msg:
                raise knext.InvalidParametersError(error_msg) from e
            LOGGER.warning(f"Error while creating model: {error_msg}")
            raise knext.InvalidParametersError(
                "Could not create model. Please provide a model in GGUF format."
            )


gpt4all_llm_port_type = knext.port_type(
    "GPT4All LLM",
    GPT4AllLLMPortObject,
    GPT4AllLLMPortObjectSpec,
    id="org.knime.python.llm.models.gpt4all.GPT4AllLLMPortObject",
)


@knext.node(
    "Local GPT4All LLM Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    category=gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local Large Language Model",
    ],
)
@knext.output_port(
    "GPT4All LLM",
    "A GPT4All large language model.",
    gpt4all_llm_port_type,
)
class GPT4AllLLMConnector:
    """
    Connects to a locally installed GPT4All LLM.

    This node allows you to connect to a local GPT4All LLM. To get started,
    you need to download a specific model either through the [GPT4All client](https://gpt4all.io)
    or by dowloading a GGUF model from [Hugging Face Hub](https://huggingface.co/).
    Once you have downloaded the model, specify its file path in the
    configuration dialog to use it.

    It is not necessary to install the GPT4All client to execute the node.

    Some models (e.g. Llama 2) have been fine-tuned for chat applications,
    so they might behave unexpectedly if their prompts do not follow a chat like structure:

        User: <The prompt you want to send to the model>
        Assistant:

    Use the prompt template for the specific model from the
    [GPT4All model list](https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models2.json)
    if one is provided.

    The currently supported models are based on GPT-J, LLaMA, MPT, Replit, Falcon and StarCoder.

    For more information and detailed instructions on downloading compatible models, please visit the [GPT4All GitHub repository](https://github.com/nomic-ai/gpt4all).

    **Note**: This node cannot be used on the KNIME Hub, as the models cannot be embedded into the workflow due to their large size.
    """

    settings = GPT4AllInputSettings()
    params = GPT4AllModelParameterSettings(since_version="5.2.0")

    def configure(self, ctx: knext.ConfigurationContext) -> GPT4AllLLMPortObjectSpec:
        is_valid_model(self.settings.local_path)
        return self.create_spec()

    def execute(self, ctx: knext.ExecutionContext) -> GPT4AllLLMPortObject:
        return GPT4AllLLMPortObject(self.create_spec())

    def create_spec(self) -> GPT4AllLLMPortObjectSpec:
        n_threads = None if self.settings.n_threads == 0 else self.settings.n_threads

        return GPT4AllLLMPortObjectSpec(
            local_path=self.settings.local_path,
            n_threads=n_threads,
            temperature=self.params.temperature,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
            max_token=self.params.max_token,
            n_ctx=self.params.n_ctx,
            prompt_batch_size=self.params.prompt_batch_size,
            device=self.params.device,
        )
