import knime.extension as knext
from models.base import LLMPortObject

from typing import Sequence, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage, SystemMessage
from giskard.llm.client.base import LLMClient


class KnimeLLMClient(LLMClient):
    def __init__(self, port_object: LLMPortObject, ctx: knext.ExecutionContext):
        self._model = port_object
        self._ctx = ctx

        # Initialize attributes if they exist in the model spec
        attributes = ["_temperature", "_max_tokens", "_caller_id", "_seed", "_format"]
        for attr in attributes:
            if hasattr(self._model.spec, attr):
                setattr(self, attr, getattr(self._model.spec, attr))

    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        """Prompts the model to generate domain-specific probes. Uses the parameters of the model instead of
        the function parameters."""

        # Update model spec attributes if provided
        attributes = {
            "_temperature": temperature,
            "_max_tokens": (
                max_tokens
                if max_tokens is not None
                else getattr(self, "_max_tokens", None)
            ),
            "_caller_id": (
                caller_id
                if caller_id is not None
                else getattr(self, "_caller_id", None)
            ),
            "_seed": seed if seed is not None else getattr(self, "_seed", None),
            "_format": format if format is not None else getattr(self, "_format", None),
        }

        for attr, value in attributes.items():
            if hasattr(self._model.spec, attr):
                setattr(self._model.spec, attr, value)

        # Create model
        model = self._model.create_model(self._ctx)

        converted_messages = self._convert_messages(messages)
        answer = model.invoke(converted_messages)
        if isinstance(model, BaseChatModel):
            answer = answer.content
        return ChatMessage(role="assistant", content=answer)

    _role_to_message_type = {
        "ai": AIMessage,
        "assistant": AIMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "system": SystemMessage,
    }

    def _create_message(self, role: str, content: str):
        if not role:
            raise RuntimeError("Giskard did not specify a message role.")
        message_type = self._role_to_message_type.get(role.lower(), None)
        if message_type:
            return message_type(content=content)
        else:
            # fallback
            return ChatMessage(content=content, role=role)

    def _convert_messages(self, messages: Sequence[ChatMessage]):
        return [self._create_message(msg.role, msg.content) for msg in messages]
