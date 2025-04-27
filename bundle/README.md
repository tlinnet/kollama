# Bundling Python extension to share a zipped update site 
See [Option 1: Bundling a Python extension to share a zipped update site ](https://docs.knime.com/latest/pure_python_node_extensions_guide/?pk_vid=0ad752aaaefd276d174488923381293f#extension-bundling)

Installed the environment via `miniforge3 Prompt`.
```powershell
cd C:/KNIME/kollama/bundle
conda env create -f env.yml
# conda env update --name knime-ext-bundling -f env.yml --prune
```

## Build
Activate environment
```powershell
cd C:/KNIME/kollama/bundle
conda activate knime-ext-bundling

# Can Takes 5-10 min to bundle.
#build_python_extension.bat <path/to/directoryof/myextension/> <path/to/directoryof/output>
build_python_extension.bat .. ./build/kollama
```

With the environment activated, run the following command to bundle the Python extensions.

## Import

Add the generated repository folder to KNIME AP with
* Add Software Site in `File -> Preferences -> Install/Update -> Available Software Sites`
  * Name: `Kollama`, Location: `file:/C:/KNIME/kollama/bundle/build/kollama/`
* Install it via `File -> Install KNIME Extensions`
  * Search for: `KNIME Ollama`