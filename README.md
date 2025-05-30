# Kollama
![kollama is a KNIME Ollama Extension](icons/kollama.png)

KNIME Ollama Extension, developed in Python. Introducing embedding node.

![image](https://github.com/user-attachments/assets/73edc4cb-fb99-480f-87bb-9d6232510c96)

Since the "KNIME AI Extension" LLM ports is not KNIME core PortTypes, the "KNIME AI Extension" is bundled with this extension.
If LLM ports will become KNIME core PortTypes, this extension will only feature Ollama nodes.

# Install Kollama - KNIME Ollama Extension

`Kollama` extension has been requesed to be published via [KNIME Community Hub](https://docs.knime.com/latest/pure_python_node_extensions_guide/#extension-bundling).

Until available in KNIME Community Hub, `Kollama` can be installed by downloading the zipped local KNIME Update site from [Kollama releases](https://github.com/tlinnet/kollama/releases).

Unzip `kollama.zip` to for example `C:/KNIME/local-extensions/kollama/`. Then add folder as a KNIME Update Site and install.

* Add Software Site in File -> Preferences -> Install/Update -> Available Software Sites
  * Name: `Kollama`, Location: file:/C:/KNIME/local-extensions/kollama/
* Install it via File -> Install KNIME Extensions
  * Search for: `KNIME Ollama`

# Development

Developed as described here 
* [4 steps for your Python Team to develop KNIME nodes](https://www.knime.com/blog/4-steps-for-your-python-team-to-develop-knime-nodes)
* [Create a New Python based KNIME Extension](https://docs.knime.com/latest/pure_python_node_extensions_guide).

## Preparation
* [Install miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) for conda environment handling
* Install KNIME Extension: [KNIME Python Extension Development (Labs)](https://hub.knime.com/knime/extensions/org.knime.features.python3.nodes/latest)

Create the folder: `C:/KNIME`. Check out the code in this folder, to `C:/KNIME/kollama`.

Resolve conda environment via `miniforge3 Prompt`.
```powershell
cd C:/KNIME/kollama
conda env create -f env.yml
# conda env update --name kollama -f env.yml --prune
```

In `knime_extension_config.yml`, update the path to `conda_env_path` to your local environment.

In `%AppData%/../Local/Programs/KNIME/knime.ini`, add path to `knime_extension_config.yml` with command:
```
-Dknime.python.extension.config=C:/KNIME/kollama/knime_extension_config.yml
```

Then start KNIME, and in `Node Repository` find `Community Nodes/kollama`.

## Execute test

Use `miniforge3 Prompt`
```
cd C:/KNIME/kollama
conda activate kollama
python -m unittest discover tests
# Or Test Single file / Class / Function
python -m unittest tests.test_models_ollama
```

## org.knime.python.llm code
Code for the KNIME extension: [KNIME AI Extension](https://hub.knime.com/knime/extensions/org.knime.python.features.llm/latest)

After installation of extension, the source code was found here.
```cmd
%AppData%/../Local/Programs/KNIME/plugins/org.knime.python.llm_5.4.3.v202503051153
```

In `KNIME Analytics Platform` -> `Help` -> `About...` -> `Installation Details`: Searching for `KNIME AI Extension`, class is `org.knime.python.features.llm.feature.group`

### Changes

`src/main/python` copied to llm_5_4_3_v202503051153

* Deleted `knime.p12`, `cyclonedx-linux-x64` ~60MB linux binary, `scripts` folder, `versions` folder

Modified
* `src/util`: Imported `main_category` from kollama
* `icons`: Copied to `icons` in kollama
* `src/util/base`: `from kollama.models._base import model_category, ChatModelPortObject, ChatModelPortObjectSpec, LLMPortObject, LLMPortObjectSpec, EmbeddingsPortObject, EmbeddingsPortObjectSpec`
  * Deleted corresponding classes. This has to be done, to let ports work together.
