from langchain_core.language_models import LLM, SimpleChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from typing import Any, List, Optional, Mapping, Dict
from pydantic import BaseModel
import knime.extension as knext
from util import MissingValueHandlingOptions


def generate_response(
    response_dict: dict[str, str],
    default_response: str,
    prompt: str,
    missing_value_strategy: str,
    node: str,
):
    response = response_dict.get(prompt)

    if not response:
        if missing_value_strategy == MissingValueHandlingOptions.Fail.name:
            raise knext.InvalidParametersError(
                f"""Could not find matching response for prompt: '{prompt}'. Please ensure that the prompt 
                exactly matches one specified in the prompt column of the {node} upstream."""
            )
        else:
            return default_response

    return response


class TestDictLLM(LLM):
    """Self implemented Test LLM wrapper for testing purposes."""

    response_dict: dict[str, str]
    default_response: str
    missing_value_strategy: str

    @property
    def _llm_type(self) -> str:
        return "test-dict"

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test LLM Connector",
        )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test LLM Connector",
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class TestChatModel(SimpleChatModel):
    """Test ChatModel for testing purposes."""

    response_dict: Dict[str, str]
    default_response: str
    missing_value_strategy: str

    @property
    def _llm_type(self) -> str:
        return "test-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = messages[len(messages) - 1].content
        return generate_response(
            self.response_dict,
            self.default_response,
            prompt,
            self.missing_value_strategy,
            "Test Chat Model Connector",
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class TestEmbeddings(Embeddings, BaseModel):
    embeddings_dict: dict[str, list[float]]
    fail_on_mismatch: bool

    def embed_documents(self, documents: List[str]) -> List[float]:
        return [self.embed_query(document) for document in documents]

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.embeddings_dict[text]
        except KeyError:
            if self.fail_on_mismatch:
                raise knext.InvalidParametersError(
                    f"""Could not find document '{text}' in the Test Embeddings Model. Please ensure that 
                    the query exactly matches one of the embedded documents."""
                )
            else:
                embeddings_dimension = len(next(iter(self.embeddings_dict.values())))
                zero_vector = [0.0 for _ in range(embeddings_dimension)]

                return zero_vector
