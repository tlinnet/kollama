# KNIME / own imports
from typing import Optional

import knime.extension as knext
from ..base import (
    model_category,
    GeneralRemoteSettings,
)


hf_icon = "icons/huggingface.png"
hf_category = knext.category(
    path=model_category,
    level_id="huggingface",
    name="Hugging Face",
    description="",
    icon=hf_icon,
)


class HFModelSettings(GeneralRemoteSettings):
    top_k = knext.IntParameter(
        label="Top k",
        description="The number of top-k tokens to consider when generating text.",
        default_value=1,
        min_value=0,
        is_advanced=True,
    )

    typical_p = knext.DoubleParameter(
        label="Typical p",
        description="The typical probability threshold for generating text.",
        default_value=0.95,
        max_value=1.0,
        min_value=0.1,
        is_advanced=True,
    )

    repetition_penalty = knext.DoubleParameter(
        label="Repetition penalty",
        description="The repetition penalty to use when generating text.",
        default_value=1.0,
        min_value=0.0,
        max_value=100.0,
        is_advanced=True,
    )

    max_new_tokens = knext.IntParameter(
        label="Max new tokens",
        description="""
        The maximum number of tokens to generate in the completion.

        The token count of your prompt plus *max new tokens* cannot exceed the model's context length.
        """,
        default_value=50,
        min_value=0,
    )


@knext.parameter_group(label="Prompt Templates")
class HFPromptTemplateSettings:
    system_prompt_template = knext.MultilineStringParameter(
        "System prompt template",
        """Model specific system prompt template. Defaults to "%1".
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )

    prompt_template = knext.MultilineStringParameter(
        "Prompt template",
        """Model specific prompt template. Defaults to "%1". 
        Refer to the Hugging Face Hub model card for information on the correct prompt template.""",
        default_value="%1",
    )


def raise_for(exception: Exception, default: Optional[Exception] = None):
    import requests

    if isinstance(exception, requests.exceptions.ProxyError):
        raise RuntimeError(
            "Failed to establish connection due to a proxy error. Validate your proxy settings."
        ) from exception
    if isinstance(exception, requests.exceptions.Timeout):
        raise RuntimeError(
            "The connection to Hugging Face Hub timed out."
        ) from exception
    if default:
        raise default from exception
    raise exception
