import knime.extension as knext
import util
from dataclasses import dataclass

tortoise_icon = "icons/tortoise_icon.png"

eval_category = knext.category(
    path=util.main_category,
    level_id="evaluation",
    name="Evaluation",
    description="",
    icon=tortoise_icon,
)


def _get_schema_from_workflow_spec(
    workflow_spec, return_input_schema: bool
) -> knext.Schema:
    if workflow_spec is None:
        raise knext.InvalidParametersError(
            "Workflow spec is not available. Execute predecessor nodes."
        )
    if return_input_schema:
        return next(iter(workflow_spec.inputs.values())).schema
    else:
        return next(iter(workflow_spec.outputs.values())).schema


def _get_workflow_schema(
    ctx: knext.DialogCreationContext, port: int, input: bool
) -> knext.Schema:
    return _get_schema_from_workflow_spec(ctx.get_input_specs()[port], input)


def _validate_prediction_workflow_spec(workflow_spec) -> None:
    if len(workflow_spec.inputs) != 1:
        raise knext.InvalidParametersError(
            "Prediction workflow must have exactly one input table."
        )

    if len(workflow_spec.outputs) != 1:
        raise knext.InvalidParametersError(
            "Prediction workflow must produce exactly one output table."
        )


def _pick_default_workflow_column(workflow_spec, input: bool) -> str:
    prediction_workflow_schema = _get_schema_from_workflow_spec(
        workflow_spec, return_input_schema=input
    )
    return util.pick_default_column(prediction_workflow_schema, knext.string())


@dataclass
class ScannerColumn:
    name: str
    knime_type: knext.KnimeType
    pd_type: type
