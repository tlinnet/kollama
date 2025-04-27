from typing import List, Optional
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import knime.api.schema as ks


class OpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._model = model
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, default_headers=extra_headers
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [embedding.embedding for embedding in self._embed(texts)]

    def embed_query(self, text: str) -> ks.List[float]:
        return self._embed(text)[0].embedding

    def _embed(self, input: str | List[str]) -> List:
        return self._client.embeddings.create(
            model=self._model,
            input=input,
            encoding_format="float",
        ).data
