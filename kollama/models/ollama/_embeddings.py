import knime.extension as knext
# These import has to be relative. When using "src.models.x" the nodes disappear in the KNIME GUI.
from .._base import EmbeddingsPortObjectSpec, EmbeddingsPortObject
from ._auth import OllamaAuthenticationPortObjectSpec
from ._embeddings_model import OllamaEmbeddings


class OllamaEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(
        self,
        auth: OllamaAuthenticationPortObjectSpec,
        model: str,
    ) -> None:
        super().__init__()
        self._auth = auth
        self._model = model

    @property
    def auth(self) -> OllamaAuthenticationPortObjectSpec:
        return self._auth

    @property
    def model(self) -> str:
        return self._model

    def validate_context(self, ctx):
        self.auth.validate_context(ctx)

    def serialize(self) -> dict:
        return {
            "auth": self.auth.serialize(),
            "model": self.model,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            auth=OllamaAuthenticationPortObjectSpec.deserialize(data["auth"]),
            model=data["model"],
        )


class OllamaEmbeddingsPortObject(EmbeddingsPortObject):
    @property
    def spec(self) -> OllamaEmbeddingsPortObjectSpec:
        return super().spec

    def create_model(self, 
        ctx: knext.ExecutionContext):

        return OllamaEmbeddings(
            base_url=self.spec.auth.base_url,
            timeout=self.spec.auth.timeout,
            model=self.spec.model,
        )