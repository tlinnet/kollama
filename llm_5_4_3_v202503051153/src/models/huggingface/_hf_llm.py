from typing import Any, List, Optional
from langchain_core.language_models import LLM
from pydantic import model_validator
from huggingface_hub import InferenceClient
from .hf_base import raise_for
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import util


class HFLLM(LLM):
    """Custom implementation backed by huggingface_hub.InferenceClient.
    We can't use the implementation of langchain_community because it always requires an api token (and is
    probably going to be deprecated soon) and we also can't use the langchain_huggingface implementation
    since it has torch as a required dependency."""

    model: str
    """Can be a repo id on hugging face hub or the url of a TGI server."""
    hf_api_token: Optional[str] = None
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    client: Any
    seed: Optional[int] = None

    def _llm_type(self):
        return "hfllm"

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: dict) -> dict:
        values["client"] = InferenceClient(
            model=values["model"],
            timeout=120,
            token=values.get("hf_api_token"),
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        client: InferenceClient = self.client
        try:
            return client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                typical_p=self.typical_p,
                seed=self.seed,
            )
        except Exception as ex:
            raise_for(ex)


class HuggingFaceEmbeddings(HuggingFaceHubEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceHubEmbeddings 'embed_documents' to allow batches
        return util.batched_apply(super().embed_documents, texts, self.batch_size)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        # Overrides HuggingFaceHubEmbeddings 'aembed_documents' to allow batches
        return util.abatched_apply(super().aembed_documents, texts, self.batch_size)
