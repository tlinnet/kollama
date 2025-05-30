import unittest
from unittest.mock import MagicMock, patch
from ollama._types import EmbedResponse, ListResponse, ModelDetails, ProcessResponse, ResponseError, ShowResponse
import datetime

from kollama.models.ollama._auth import OllamaAuthenticationPortObjectSpec
from kollama.models.ollama._chat import OllamaChatModelPortObject
from kollama.models.ollama._embeddings_model import OllamaEmbeddings
from kollama.models.ollama._util import OllamaUtil

mock_models = MagicMock()
mock_models_listresponse = ListResponse(models=[
    ListResponse.Model(model='gemma3:12b', modified_at=datetime.datetime(2025, 3, 17), digest='6fd036cefda5093cc827b6c16be5e447f23857d4a472ce0bdba0720573d4dcd9', size=8149190199, details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='12.2B', quantization_level='Q4_K_M')), 
    ListResponse.Model(model='qwen2.5-coder:32b', modified_at=datetime.datetime(2025, 3, 8), digest='4bd6cbf2d094264457a17aab6bd6acd1ed7a72fb8f8be3cfb193f63c78dd56df', size=19851349856, details=ModelDetails(parent_model='', format='gguf', family='qwen2', families=['qwen2'], parameter_size='32.8B', quantization_level='Q4_K_M')), 
    ListResponse.Model(model='mxbai-embed-large:latest', modified_at=datetime.datetime(2025, 1, 27), digest='468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8', size=669615493, details=ModelDetails(parent_model='', format='gguf', family='bert', families=['bert'], parameter_size='334M', quantization_level='F16')), 
    ListResponse.Model(model='nomic-embed-text:latest', modified_at=datetime.datetime(2025, 1, 27), digest='0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f', size=274302450, details=ModelDetails(parent_model='', format='gguf', family='nomic-bert', families=['nomic-bert'], parameter_size='137M', quantization_level='F16'))
])
mock_models.list.return_value = mock_models_listresponse

mock_models_showresponse = ShowResponse(
    modified_at=datetime.datetime(2025, 3, 17), 
    template=r'{{- range $i, $_ := .Messages }}\n{{- $last := eq (len (slice $.Messages $i)) 1 }}\n{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user\n{{ .Content }}<end_of_turn>\n{{ if $last }}<start_of_turn>model\n{{ end }}\n{{- else if eq .Role "assistant" }}<start_of_turn>model\n{{ .Content }}{{ if not $last }}<end_of_turn>\n{{ end }}\n{{- end }}\n{{- end }}',
    modelfile=r'# Modelfile generated by "ollama show"\n# To build a new Modelfile based on this, replace FROM with:\n# FROM gemma3:12b\n\nFROM ...', 
    license=r'Gemma Terms of Use \n\nLast modified: February 21, 2024\n\nBy using, reproducing, modifying, distributing, performing or displaying any portion or element of Gemma, Model Derivatives including via any Hosted Service ...', 
    details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='12.2B', quantization_level='Q4_K_M'),
    model_info={'gemma3.attention.head_count': 16, 'gemma3.attention.head_count_kv': 8, 'gemma3.attention.key_length': 256, 'gemma3.attention.sliding_window': 1024, 'gemma3.attention.value_length': 256, 'gemma3.block_count': 48, 'gemma3.context_length': 8192, 'gemma3.embedding_length': 3840, 'gemma3.feed_forward_length': 15360, 'gemma3.vision.attention.head_count': 16, 'gemma3.vision.attention.layer_norm_epsilon': 1e-06, 'gemma3.vision.block_count': 27, 'gemma3.vision.embedding_length': 1152, 'gemma3.vision.feed_forward_length': 4304, 'gemma3.vision.image_size': 896, 'gemma3.vision.num_channels': 3, 'gemma3.vision.patch_size': 14, 'general.architecture': 'gemma3', 'general.file_type': 15, 'general.parameter_count': 12187079280, 'general.quantization_version': 2, 'tokenizer.ggml.add_bos_token': True, 'tokenizer.ggml.add_eos_token': False, 'tokenizer.ggml.add_padding_token': False, 'tokenizer.ggml.add_unknown_token': False, 'tokenizer.ggml.bos_token_id': 2, 'tokenizer.ggml.eos_token_id': 1, 'tokenizer.ggml.merges': None, 'tokenizer.ggml.model': 'llama', 'tokenizer.ggml.padding_token_id': 0, 'tokenizer.ggml.pre': 'default', 'tokenizer.ggml.scores': None, 'tokenizer.ggml.token_type': None, 'tokenizer.ggml.tokens': None, 'tokenizer.ggml.unknown_token_id': 3},
    parameters=r'stop                           "<end_of_turn>"\ntemperature                    0.1'
)

mock_models_psresponse = ProcessResponse(models=[
    ProcessResponse.Model(model='nomic-embed-text:latest', name='nomic-embed-text:latest', digest='0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f', expires_at=datetime.datetime(2025, 4, 28, 12, 36, 46), size=849202176, size_vram=849202176, details=ModelDetails(parent_model='', format='gguf', family='nomic-bert', families=['nomic-bert'], parameter_size='137M', quantization_level='F16')), 
    ProcessResponse.Model(model='gemma3:12b', name='gemma3:12b', digest='6fd036cefda5093cc827b6c16be5e447f23857d4a472ce0bdba0720573d4dcd9', expires_at=datetime.datetime(2025, 4, 28, 12, 36, 46), size=24564058048, size_vram=24564058048, details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='12.2B', quantization_level='Q4_K_M')), 
])


class TestOllamaAuth(unittest.TestCase):
    def setUp(self):
        # Set up a mock context and credentials
        self.mock_ctx = MagicMock()
        self.spec = OllamaAuthenticationPortObjectSpec(base_url="http://localhost:11434")

    @patch('kollama.models.ollama._util.Client')
    def test_get_models_success(self, mock_ollama_client):
        mock_ollama_client.return_value = mock_models
        # Call the method
        models = self.spec._get_models_from_api(self.mock_ctx)
        assert isinstance(models, list)
        assert len(models) > 0

    @patch("kollama.models.ollama._util.Client")
    def test_get_models_404_not_found(self, mock_ollama_client):
        mock_response = MagicMock()
        mock_response.list.side_effect = ResponseError("HTTP Error 404. The requested resource is not found")
        mock_ollama_client.return_value = mock_response
        # Call the method
        self.spec._base_url = "http://localhost"
        with self.assertRaises(ResponseError) as context:
            self.spec._get_models_from_api(self.mock_ctx)


class TestOllamaChat(unittest.TestCase):
    def setUp(self):
        # Set up a mock context and credentials
        self.mock_ctx = MagicMock()
        mock_spec = MagicMock()
        self.portobj = OllamaChatModelPortObject(spec=mock_spec)
        self.portobj.spec.auth.base_url = "http://localhost:11434"
        self.portobj.spec.auth.timeout = 2

    @patch("kollama.models.ollama._chat.ChatOllama")
    def test_create_model_standard(self, mock_chat_openai):
        # Setup mock spec
        self.portobj.spec.model = "normal-model"
        self.portobj.spec.temperature = 0.7
        self.portobj.spec.num_predict = 1000
        # Call method
        self.portobj.create_model(self.mock_ctx)
        # Verify ChatOllama was called with correct params
        mock_chat_openai.assert_called_once_with(
            base_url="http://localhost:11434",
            model="normal-model",
            temperature=0.7,
            num_predict=1000,
            timeout=2
        )

    @patch("kollama.models.ollama._chat.ChatOllama")
    def test_create_model_reasoner(self, mock_chat_openai):
        # Setup mock spec
        self.portobj.spec.model = "reasoner-model"
        self.portobj.spec.num_predict = 1000
        # Call method 
        self.portobj.create_model(self.mock_ctx)
        # Verify ChatOllama was called with correct params for reasoner
        mock_chat_openai.assert_called_once_with(
            base_url="http://localhost:11434",
            model="reasoner-model", 
            temperature=1,
            num_predict=1000,
            timeout=2
        )


class TestOllamaEmbeddings(unittest.TestCase):
    def setUp(self):
        self.model = "nomic-embed-text:latest"
        self.base_url = "http://localhost:11434"
        self.embeddings = OllamaEmbeddings(model=self.model, base_url=self.base_url, timeout=2)
        # Mock
        self.mock_response = MagicMock()
        self.mock_response.embed.return_value = EmbedResponse(model='nomic-embed-text:latest', created_at=None, done=None, done_reason=None,
            total_duration=133535684, load_duration=1526924, prompt_eval_count=2, prompt_eval_duration=None, eval_count=None, eval_duration=None,
            embeddings=[[0.1, 0.2, 0.3]]
        )
        # Mock 2
        self.mock_response2 = MagicMock()
        self.mock_response2.embed.return_value = EmbedResponse(model='nomic-embed-text:latest', created_at=None, done=None, done_reason=None,
            total_duration=184507360, load_duration=780584, prompt_eval_count=6, prompt_eval_duration=None, eval_count=None, eval_duration=None,
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

    @patch("kollama.models.ollama._embeddings_model.Client")
    def test__embed_str(self, mock_ollama_client):
        mock_ollama_client.return_value = self.mock_response
        # Call the method
        text = "test text"
        embed_text = self.embeddings._embed(input=text)
        assert isinstance(embed_text, EmbedResponse)
        assert isinstance(embed_text.embeddings, list)
        assert len(embed_text.embeddings) == 1

    @patch("kollama.models.ollama._embeddings_model.Client")
    def test__embed_list_str(self, mock_ollama_client):
        mock_ollama_client.return_value = self.mock_response2
        # Call the method
        text_list = ["test text1", "test text2"]
        embed_text_list = self.embeddings._embed(text_list)
        assert isinstance(embed_text_list, EmbedResponse)
        assert isinstance(embed_text_list.embeddings, list)
        assert len(embed_text_list.embeddings) == 2

    @patch("kollama.models.ollama._embeddings_model.Client")
    def test_embed_query(self, mock_ollama_client):
        mock_ollama_client.return_value = self.mock_response
        # Call the method
        query = "test text"
        embed_query = self.embeddings.embed_query(text=query)
        assert isinstance(embed_query, list)
        assert len(embed_query) == 3

    @patch("kollama.models.ollama._embeddings_model.Client")
    def test_embed_documents(self, mock_ollama_client):
        mock_ollama_client.return_value = self.mock_response2
        # Call the method
        docs = ["test text1", "test text2"]
        embed_docs = self.embeddings.embed_documents(docs)
        assert isinstance(embed_docs, list)
        assert len(embed_docs) == 2
        assert len(embed_docs[0]) == 3


class TestOllamaUtil(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:11434"
        self.util = OllamaUtil(base_url=self.base_url, timeout=2)

    def fake_ollama_list(self):
        return mock_models_listresponse

    def fake_ollama_show(self, model=None):
        return mock_models_showresponse

    def fake_ollama_ps(self):
        return mock_models_psresponse

    @patch.object(OllamaUtil, 'ollama_list', fake_ollama_list)
    def test_ollama_list(self):
        # Call the method
        ollama_list = self.util.ollama_list()
        assert isinstance(ollama_list, ListResponse)
        assert len(ollama_list.models) == 4

    @patch.object(OllamaUtil, 'ollama_show', fake_ollama_show)
    def test_ollama_show(self):
        # Call the method
        ollama_show = self.util.ollama_show(model="gemma3:12b")
        assert isinstance(ollama_show, ShowResponse)
        assert ollama_show.modelinfo['gemma3.context_length'] == 8192

    @patch.object(OllamaUtil, 'ollama_list', fake_ollama_list)
    def test_ollama_list_models(self):
        # Call the method
        models = self.util.ollama_list_models(verbose=False)
        assert isinstance(models, dict)
        assert len(models) == 4
        assert set(models.keys()) == set( ['gemma3:12b', 'qwen2.5-coder:32b', 'mxbai-embed-large:latest','nomic-embed-text:latest'])

    @patch.object(OllamaUtil, 'ollama_list', fake_ollama_list)
    def test_ollama_list_models_chat(self):
        # Call the method
        models = self.util.ollama_list_models(mode="chat")
        assert isinstance(models, dict)
        assert len(models) == 2
        assert set(models.keys()) == set(['gemma3:12b', 'qwen2.5-coder:32b'])

    @patch.object(OllamaUtil, 'ollama_ps', fake_ollama_ps)
    def test_ollama_ps(self):
        # Call the method
        ollama_ps = self.util.ollama_ps()
        assert isinstance(ollama_ps, ProcessResponse)
        assert len(ollama_ps.models) == 2

    @patch.object(OllamaUtil, 'ollama_ps', fake_ollama_ps)
    def test_ollama_ps_models(self):
        # Call the method
        models = self.util.ollama_ps_models()
        assert isinstance(models, dict)
        assert len(models) == 2
        assert set(models.keys()) == set( ['nomic-embed-text:latest', 'gemma3:12b'])
