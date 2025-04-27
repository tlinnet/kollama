import knime.extension as knext
from ..base import model_category
import knime.api.schema as ks
from urllib.parse import urlparse, urlunparse
from typing import Callable

hub_connector_icon = "icons/Hub_AI_connector.png"

knime_category = knext.category(
    path=model_category,
    level_id="knime",
    name="KNIME",
    description="Models that connect to the KNIME Hub.",
    icon=hub_connector_icon,
)


def create_authorization_headers(
    auth_spec: ks.HubAuthenticationPortObjectSpec,
) -> dict[str, str]:
    return {
        "Authorization": f"{auth_spec.auth_schema} {auth_spec.auth_parameters}",
    }


def validate_auth_spec(auth_spec: ks.HubAuthenticationPortObjectSpec) -> None:
    if auth_spec.hub_url is None:
        raise knext.InvalidParametersError(
            "KNIME Hub connection not available. Please re-execute the node."
        )


def extract_api_base(auth_spec: ks.HubAuthenticationPortObjectSpec) -> str:
    try:
        validate_auth_spec(auth_spec)
    except knext.InvalidParametersError as ex:
        # ValueError does not add the exception type to the error message in the dialog
        raise ValueError(str(ex))
    hub_url = auth_spec.hub_url
    parsed_url = urlparse(hub_url)
    # drop params, query and fragment
    ai_proxy_url = (parsed_url.scheme, parsed_url.netloc, "ai-gateway/v1", "", "", "")
    return urlunparse(ai_proxy_url)


def create_model_choice_provider(
    mode: str,
) -> Callable[[knext.DialogCreationContext], list[str]]:
    def model_choices_provider(ctx: knext.DialogCreationContext) -> list[str]:
        model_list = []
        if (specs := ctx.get_input_specs()) and (auth_spec := specs[0]):
            model_list = list_models(auth_spec, mode)
            model_list.sort()
        return model_list

    return model_choices_provider


def list_models(auth_spec, mode: str) -> list[str]:
    return [model_data["name"] for model_data in _get_model_data(auth_spec, mode)]


def _get_model_data(auth_spec, mode: str):
    import requests

    api_base = extract_api_base(auth_spec)
    model_info = api_base + "/management/models?mode=" + mode
    response = requests.get(
        url=model_info, headers=create_authorization_headers(auth_spec)
    )
    if response.status_code == 404:
        raise ValueError(
            "The GenAI gateway is not reachable. Is it activated in the connected KNIME Hub?"
        )
    response.raise_for_status()
    return response.json()["models"]


def list_models_with_descriptions(auth_spec, mode: str) -> list[tuple[str, str, str]]:
    return [
        (data.get("name"), data.get("mode"), data.get("description") or None)
        for data in _get_model_data(auth_spec, mode)
    ]
