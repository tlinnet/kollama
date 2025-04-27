import knime.extension as knext
from ..base import model_category

deepseek_icon = "icons/deepseek.png"
deepseek_category = knext.category(
    path=model_category,
    name="DeepSeek",
    level_id="deepseek",
    description="DeepSeek models",
    icon=deepseek_icon,
)
