from ollama import Client
from ollama._types import ListResponse, ProcessResponse, ShowResponse
import knime.extension as knext

from kollama.models._base import model_category

ollama_icon = "icons/ollama.png"
ollama_category = knext.category(
    path=model_category,
    name="Ollama",
    level_id="ollama",
    description="Ollama models",
    icon=ollama_icon,
)


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

class OllamaUtil():
    def __init__(self,
        base_url: str,
        timeout: int
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._client = Client(host=self._base_url, timeout=self._timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def ollama(self) -> Client:
        return self._client

    def ollama_list(self) -> ListResponse:
        ollama_list = self.ollama.list()
        return ollama_list

    def ollama_show(self, model: str) -> ShowResponse:
        ollama_show = self.ollama.show(model=model)
        return ollama_show

    def ollama_list_models(self, mode: str | None = None, verbose: bool = False) -> dict:
        models = {}
        ollama_list = self.ollama_list().models
        for model in ollama_list:
            model_type = 'other'
            if model.details.quantization_level.startswith("Q"):
                model_type = 'chat'
            elif model.details.quantization_level.startswith("F"):
                model_type = 'embedding'

            d = {'type':model_type, 'size':sizeof_fmt(model.size), 'quantization_level':model.details.quantization_level, 'family':model.details.family, 'parameter_size=' :model.details.parameter_size, 
                 'context_length':None, 'embedding_length':None}

            if mode and model_type == mode:
                models[model.model] = d
            if mode and model_type != mode:
                pass
            else:
                models[model.model] = d

        if verbose:
            for model in models:
                ollama_show = self.ollama_show(model=model)
                for mi in ollama_show.modelinfo:
                    if 'context_length' in mi:
                        models[model]['context_length'] = ollama_show.modelinfo.get(mi)
                    elif 'embedding_length' in mi:
                        models[model]['embedding_length'] = ollama_show.modelinfo.get(mi)

        return models

    def ollama_ps(self) -> ProcessResponse:
        return self.ollama.ps()