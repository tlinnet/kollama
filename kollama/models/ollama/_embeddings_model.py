from typing import List
from langchain_core.embeddings import Embeddings
from ollama import Client
from ollama._types import EmbedResponse
import knime.api.schema as ks


class OllamaEmbeddings(Embeddings):
    def __init__(self,
        base_url: str,
        timeout: int,
        model: str,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._model = model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts).embeddings

    def embed_query(self, text: str) -> ks.List[float]:
        return self._embed(text).embeddings[0]

    def _embed(self, input: str | List[str]) -> EmbedResponse:
        ollama = Client(host=self.base_url, timeout=self.timeout)
        models_response =  ollama.embed(model=self.model, input=input)
        return models_response
