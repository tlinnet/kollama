import knime.extension as knext
from langchain_ollama import ChatOllama

from kollama.models._base import ChatModelPortObject, ChatModelPortObjectSpec, OutputFormatOptions
from kollama.models.ollama._auth import OllamaAuthenticationPortObjectSpec


class OllamaChatModelPortObjectSpec(ChatModelPortObjectSpec):
    """Spec of a Ollama Chat Model"""

    def __init__(
        self,
        auth: OllamaAuthenticationPortObjectSpec,
        model: str,
        temperature: float,
        num_predict: int,
        n_requests=1,
    ):
        super().__init__(n_requests)
        self._auth = auth
        self._model = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._n_requests = n_requests

    @property
    def auth(self) -> OllamaAuthenticationPortObjectSpec:
        return self._auth

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def num_predict(self) -> int:
        return self._num_predict

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self.auth.serialize(),
            "model": self.model,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "n_requests": self._n_requests,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            auth=OllamaAuthenticationPortObjectSpec.deserialize(data["auth"]),
            model=data["model"],
            temperature=data["temperature"],
            num_predict=data["num_predict"],
            n_requests=data.get("n_requests", 1),
        )


class OllamaChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> OllamaChatModelPortObjectSpec:
        return super().spec

    def create_model(self,
        ctx: knext.ExecutionContext,
        output_format: OutputFormatOptions = OutputFormatOptions.Text):
        if "reasoner" in self.spec.model:
            return ChatOllama(
                base_url=self.spec.auth.base_url,
                model=self.spec.model,
                temperature=1,
                num_predict=self.spec.num_predict,
                timeout=self.spec.auth.timeout,
            )

        return ChatOllama(
            base_url=self.spec.auth.base_url,
            model=self.spec.model,
            temperature=self.spec.temperature,
            num_predict=self.spec.num_predict,
            timeout=self.spec.auth.timeout,
        )