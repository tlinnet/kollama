# kollama
![kollama is a KNIME Ollama Extension](icons/ollama.png)

KNIME Ollama Extension, developed in Python.

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