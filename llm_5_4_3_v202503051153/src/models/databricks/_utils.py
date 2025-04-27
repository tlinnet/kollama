from typing import Literal
import knime.extension.nodes as kn
import knime.extension as knext
from ..base import model_category

databricks_workspace_port_type_id = (
    "org.knime.bigdata.databricks.workspace.port.DatabricksWorkspacePortObject"
)


def workspace_port_type_available() -> bool:
    return kn.has_port_type_for_id(databricks_workspace_port_type_id)


def get_workspace_port_type():
    return kn.get_port_type_for_id(databricks_workspace_port_type_id)


def get_base_url(databricks_workspace_spec):
    from urllib.parse import urljoin

    return urljoin(databricks_workspace_spec.workspace_url, "serving-endpoints")


def get_api_key(databricks_workspace_spec):
    return databricks_workspace_spec.auth_parameters


def check_workspace_available(databricks_workspace_spec):
    try:
        databricks_workspace_spec.auth_parameters
    except ValueError:
        raise knext.InvalidParametersError(
            "Databricks Workspace is not available. Re-execute the connector node."
        )


def get_models(
    databricks_workspace_spec, model_type: Literal["chat", "embeddings"]
) -> list[str]:
    import requests
    from urllib.parse import urljoin

    base_url = get_base_url(databricks_workspace_spec)
    api_key = get_api_key(databricks_workspace_spec)

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    serving_endpoints_url = urljoin(base_url, "api/2.0/serving-endpoints")
    res = requests.get(serving_endpoints_url, headers=headers)
    res.raise_for_status()
    models = res.json()["endpoints"]
    task = "llm/v1/" + model_type
    return [model["name"] for model in models if model["task"] == task]


def get_model_choices_provider(model_type: Literal["chat", "embeddings"]):
    def get_model_choices(ctx: knext.DialogCreationContext):
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            try:
                check_workspace_available(auth_spec)
            except knext.InvalidParametersError:
                return []

            return get_models(auth_spec, model_type)
        return []

    return get_model_choices


# looks nicer
databricks_icon = "icons/Databricks-embeddings-connector.png"

databricks_category = knext.category(
    path=model_category,
    level_id="databricks",
    name="Databricks",
    description="Contains nodes to connect models hosted on Databricks.",
    icon=databricks_icon,
)
