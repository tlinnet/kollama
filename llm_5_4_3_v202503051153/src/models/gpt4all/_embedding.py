# KNIME / own imports
import knime.extension as knext
from knime.extension.nodes import FilestorePortObject
from ..base import (
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
)

from ._utils import gpt4all_icon, gpt4all_category

from typing import Optional
import shutil
import os

_embeddings4all_model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"


def _create_embedding_model(
    model_name: str, model_path: str, n_threads: int, allow_download: bool
):
    from langchain_community.embeddings import GPT4AllEmbeddings

    return GPT4AllEmbeddings(
        model_name=model_name,
        n_threads=n_threads,
        gpt4all_kwargs={"allow_download": allow_download, "model_path": model_path},
    )


class Embeddings4AllPortObjectSpec(EmbeddingsPortObjectSpec):
    """The Embeddings4All port object spec."""

    def __init__(self, num_threads: int = 0) -> None:
        super().__init__()
        self._num_threads = num_threads

    @property
    def num_threads(self) -> int:
        return self._num_threads

    def serialize(self) -> dict:
        return {
            "num_threads": self._num_threads,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(data["num_threads"])


class Embeddings4AllPortObject(EmbeddingsPortObject, FilestorePortObject):
    """
    The Embeddings4All port object.

    The port object copies the Embeddings4All model into a filestore in order
    to make workflows containing such models portable.
    """

    def __init__(
        self,
        spec: EmbeddingsPortObjectSpec,
        model_name: str = _embeddings4all_model_name,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(spec)
        self._model_name = model_name
        self._model_path = model_path

    @property
    def spec(self) -> Embeddings4AllPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        try:
            return _create_embedding_model(
                model_name=self._model_name,
                n_threads=self.spec.num_threads,
                model_path=self._model_path,
                allow_download=False,
            )
        except Exception as e:
            unsupported_model_exception = (
                "Unable to instantiate model: Unsupported model architecture: bert"
            )
            if str(e) == unsupported_model_exception:
                raise knext.InvalidParametersError(
                    "The current embeddings model is incompatible. "
                    "Please run the GPT4All Embeddings Connector again to download the latest model, "
                    "or update it manually to a newer version. "
                    "For additional details on available models, please refer to: "
                    "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                )
            raise ValueError(f"The model at path {self.model_path} is not valid.")

    def write_to(self, file_path: str) -> None:
        from requests.exceptions import (
            ConnectionError,
        )  # The ConnectionError inherits from IOError, so we need the import

        os.makedirs(file_path)
        if self._model_path:
            # should be verified in the connector
            shutil.copy(
                os.path.join(self._model_path, self._model_name),
                os.path.join(file_path, self._model_name),
            )
        else:
            try:
                _create_embedding_model(
                    model_name=_embeddings4all_model_name,
                    model_path=file_path,
                    n_threads=1,
                    allow_download=True,
                )
            except ConnectionError:
                raise knext.InvalidParametersError(
                    "Connection error. Please ensure that your internet connection is enabled to download the model."
                )

    @classmethod
    def read_from(
        cls, spec: Embeddings4AllPortObjectSpec, file_path: str
    ) -> "Embeddings4AllPortObject":
        model_name = os.listdir(file_path)[0]
        return cls(spec, model_name, file_path)


embeddings4all_port_type = knext.port_type(
    "GPT4All Embeddings",
    Embeddings4AllPortObject,
    Embeddings4AllPortObjectSpec,
    id="org.knime.python.llm.models.gpt4all.Embeddings4AllPortObject",
)


class ModelRetrievalOptions(knext.EnumParameterOptions):
    DOWNLOAD = (
        "Download",
        "Downloads the model from GPT4All during execution. Requires an internet connection.",
    )
    READ = ("Read", "Reads the model from the local file system.")


@knext.node(
    "GPT4All Embeddings Connector",
    knext.NodeType.SOURCE,
    gpt4all_icon,
    gpt4all_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Local RAG",
        "Local Retrieval Augmented Generation",
    ],
)
@knext.output_port(
    "GPT4All Embedding model",
    "A GPT4All Embedding model that calculates embeddings on the local machine.",
    embeddings4all_port_type,
)
class Embeddings4AllConnector:
    """
    Connects to an embedding model that runs on the local machine.

    This node connects to an embedding model that runs on the local machine via GPT4All.

    The default model was trained on sentences and short paragraphs of English text.

    **Note**: Unlike the other GPT4All nodes, this node can be used on the KNIME Hub.
    """

    model_retrieval = knext.EnumParameter(
        "Model retrieval",
        "Defines how the model is retrieved during execution.",
        ModelRetrievalOptions.DOWNLOAD.name,
        ModelRetrievalOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    model_path = knext.LocalPathParameter(
        "Path to model", "The local file system path to the model."
    ).rule(
        knext.OneOf(model_retrieval, [ModelRetrievalOptions.READ.name]),
        knext.Effect.SHOW,
    )

    num_threads = knext.IntParameter(
        "Number of threads",
        """The number of threads the model uses. 
        More threads may reduce the runtime of queries to the model.

        If set to 0, the number of threads is determined automatically.""",
        0,
        min_value=0,
        is_advanced=True,
    )

    def configure(self, ctx) -> Embeddings4AllPortObjectSpec:
        return self._create_spec()

    def _create_spec(self) -> Embeddings4AllPortObjectSpec:
        n_threads = None if self.num_threads == 0 else self.num_threads
        return Embeddings4AllPortObjectSpec(n_threads)

    def execute(self, ctx) -> Embeddings4AllPortObject:
        if self.model_retrieval == ModelRetrievalOptions.DOWNLOAD.name:
            model_path = None
            model_name = _embeddings4all_model_name
        else:
            if not os.path.exists(self.model_path):
                raise ValueError(
                    f"The provided model path {self.model_path} does not exist."
                )
            model_path, model_name = os.path.split(self.model_path)
            try:
                _create_embedding_model(
                    model_name=model_name,
                    model_path=model_path,
                    n_threads=self.num_threads,
                    allow_download=False,
                )
            except Exception as e:
                unsupported_model_exception = (
                    "Unable to instantiate model: Unsupported model architecture: bert"
                )
                if str(e) == unsupported_model_exception:
                    raise knext.InvalidParametersError(
                        "The current embeddings model is incompatible. "
                        "Please run the GPT4All Embeddings Connector again to download the latest model, "
                        "or update it manually to a newer version. "
                        "For additional details on available models, please refer to: "
                        "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"
                    )
                raise ValueError(f"The model at path {self.model_path} is not valid.")

        return Embeddings4AllPortObject(
            self._create_spec(),
            model_name=model_name,
            model_path=model_path,
        )
