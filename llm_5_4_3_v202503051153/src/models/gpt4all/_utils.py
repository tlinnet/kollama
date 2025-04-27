# KNIME / own imports
import knime.extension as knext
from ..base import (
    model_category,
)

gpt4all_icon = "icons/gpt4all.png"
gpt4all_category = knext.category(
    path=model_category,
    level_id="gpt4all",
    name="GPT4All",
    description="Contains nodes for connecting to GPT4All models.",
    icon=gpt4all_icon,
)


def is_valid_model(model_path: str):
    import os

    if not model_path:
        raise knext.InvalidParametersError("Path to local model is missing")

    if not os.path.isfile(model_path):
        raise knext.InvalidParametersError(f"No file found at path: {model_path}")

    if not model_path.endswith(".gguf"):
        raise knext.InvalidParametersError(
            "Models needs to be of type '.gguf'. Find the latest models at: https://gpt4all.io/"
        )
