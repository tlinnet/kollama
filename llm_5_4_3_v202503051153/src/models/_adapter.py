from typing import List, Optional, Any, Sequence
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMChatModelAdapter(BaseChatModel):
    """
    This class adapts LLMs as chat models, allowing LLMs that have been fined tuned for
    chat applications to be used as chat models with the Chat Model Prompter.
    """

    llm: BaseLLM
    system_prompt_template: str
    prompt_template: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = self._predict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prediction = await self._apredict_messages(
            messages=messages,
            system_prompt_template=self.system_prompt_template,
            prompt_template=self.prompt_template,
            stop=stop,
            **kwargs,
        )
        return ChatResult(generations=[ChatGeneration(message=prediction)])

    def _apply_prompt_templates(
        self,
        messages: Sequence[BaseMessage],
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        string_messages = []

        message_templates = {
            HumanMessage: prompt_template,
            AIMessage: "%1",
            SystemMessage: system_prompt_template,
        }

        for m in messages:
            if type(m) not in message_templates:
                raise ValueError(f"Got unsupported message type: {m}")

            template = message_templates[type(m)]

            # if the template doesn't include the predefined pattern "%1",
            # the template input will be ignored and only the entered message will be passed
            if "%1" in template:
                message = template.replace(
                    "%1",
                    (
                        m.content
                        if isinstance(m, (HumanMessage, SystemMessage))
                        else m.content
                    ),
                )
            else:
                message = m.content

            string_messages.append(message)

        return "\n".join(string_messages)

    def _predict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self.llm(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    async def _apredict_messages(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        system_prompt_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        text = self._apply_prompt_templates(
            messages=messages,
            system_prompt_template=system_prompt_template,
            prompt_template=prompt_template,
        )
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = await self.llm._call_async(text, stop=_stop, **kwargs)
        return AIMessage(content=content)

    def _llm_type(self) -> str:
        """Return type of llm."""
        return "LLMChatModelAdapter"
