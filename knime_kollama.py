from kollama.models.ollama.auth import OllamaAuthenticator
from kollama.models.ollama.chat import OllamaChatModelConnector
from kollama.models.ollama.embeddings import OllamaEmbeddingsConnector
from kollama.models.ollama.util import OllamaModelLister

# Fix import paths
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.joinpath('llm_5_4_3_v202503051153/src')))
import llm_5_4_3_v202503051153.src.knime_llm as knime_llm