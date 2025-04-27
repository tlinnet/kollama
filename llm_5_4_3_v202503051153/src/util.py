import knime.extension as knext
from typing import Callable, List
from abc import ABC, abstractmethod
import re
import pyarrow as pa
import pyarrow.compute as pc
from dataclasses import dataclass


def is_nominal(column: knext.Column) -> bool:
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def create_type_filer(ktype: knext.KnimeType) -> Callable[[knext.Column], bool]:
    return lambda c: c.ktype == ktype


ai_icon = "icons/ml.png"

main_category = knext.category(
    path="/labs",
    level_id="kai",
    name="AI",
    description="",
    icon=ai_icon,
)


def check_canceled(ctx: knext.ExecutionContext) -> None:
    if ctx.is_canceled():
        raise RuntimeError("Execution canceled.")


def pick_default_column(input_table: knext.Schema, ktype: knext.KnimeType) -> str:
    default_column = pick_default_columns(input_table, ktype, 1)[0]
    return default_column


def pick_default_columns(
    input_table: knext.Schema, ktype: knext.KnimeType, n_columns: int
) -> List[str]:
    columns = [c for c in input_table if c.ktype == ktype]

    if len(columns) < n_columns:
        raise knext.InvalidParametersError(
            f"The input table does not contain enough ({n_columns}) distinct columns of type '{str(ktype)}'. Found: {len(columns)}"
        )

    return [column_name.name for column_name in columns[n_columns * -1 :]]


def check_column(
    input_table: knext.Schema,
    column_name: str,
    expected_type: knext.KnimeType,
    column_purpose: str,
    table_name: str = "input table",
) -> None:
    """
    Raises an InvalidParametersError if a column named column_name is not contained in input_table or has the wrong KnimeType.
    """
    if column_name not in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is missing in the {table_name}."
        )
    ktype = input_table[column_name].ktype
    if ktype != expected_type:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is of type {str(ktype)} but should be of type {str(expected_type)}."
        )


def handle_column_name_collision(
    column_names: List[str],
    column_name: str,
) -> str:
    """
    If the output column name collides with an input column name, it's been made unique by appending (#<count>).
    For example, if "column" exists as an input column name and is entered as an output column name,
    the output column name will be "column (#1)". Adding "column" or "column (#1)" again will result
    in "column1 (#2)".
    """
    basename = column_name.strip()
    existing_column_names = set(column_names)

    if basename not in existing_column_names:
        return basename

    # Pattern to match strings that have a name followed by a
    # numerical identifier in parentheses, e.g. "column (#1)"
    pattern = re.compile(r"^(.*) \(#(\d+)\)$")
    match = pattern.match(basename)
    if match:
        basename, index = match.groups()
        index = int(index) + 1
    else:
        index = 1

    new_column_name = basename
    while new_column_name in existing_column_names:
        new_column_name = f"{basename} (#{index})"
        index += 1

    return new_column_name


class MissingValueHandlingOptions(knext.EnumParameterOptions):
    SkipRow = (
        "Skip rows",
        "Rows with missing values will be ignored.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


class MissingValueOutputOptions(knext.EnumParameterOptions):
    """
    Instead of skipping rows with missing values as done with MissingValueHandlingOptions,
    this class outputs missing values. This ensures table consistency by maintaining
    the original row count, preventing the confusion caused by sudden table shrinkage
    for nodes that perform row-wise mapping operations (e.g. Text Embedder).
    """

    OutputMissingValues = (
        "Output missing values",
        "Rows with missing values will not be processed but are included in the output.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


def skip_missing_values(df, col_name: str, ctx: knext.ExecutionContext):
    import pandas as pd

    df: pd.DataFrame = df
    # Drops rows with missing values
    df_cleaned = df.dropna(subset=[col_name], how="any")
    n_skipped_rows = len(df) - len(df_cleaned)

    if n_skipped_rows > 0:
        ctx.set_warning(f"{n_skipped_rows} / {len(df)} rows are skipped.")

    return df_cleaned


def handle_missing_and_empty_values(
    df,
    input_column: str,
    missing_value_handling_setting: MissingValueHandlingOptions,
    ctx: knext.ExecutionContext,
    check_empty_values: bool = True,
):
    import pandas as pd

    df: pd.DataFrame = df
    # Drops rows if SkipRow option is selected, otherwise fails
    # if there are any missing values in the input column (=Fail option is selected)
    has_missing_values = df[input_column].isna().any()
    if (
        missing_value_handling_setting == MissingValueHandlingOptions.SkipRow
        and has_missing_values
    ):
        df = skip_missing_values(df, input_column, ctx)
    elif has_missing_values:
        missing_row_id = df[df[input_column].isnull()].index[0]
        raise ValueError(
            f"There are missing values in column {input_column}. See row ID <{missing_row_id}> for the first row that contains a missing value."
        )

    if df.empty:
        raise ValueError("All rows are skipped due to missing values.")

    if check_empty_values:
        # Check for empty values
        for id, value in df[input_column].items():
            if not value.strip():
                raise ValueError(
                    f"Empty values are not supported. See row ID {id} for the first empty value."
                )

    return df


class BaseMapper(ABC):
    def __init__(self, column: str, fn: Callable, output_type) -> None:
        super().__init__()
        self._column = column
        self._fn = fn
        self._output_type = output_type

    def _is_valid(self, text_array):
        text_array = pc.utf8_trim_whitespace(text_array)
        return pc.fill_null(pc.not_equal(text_array, pa.scalar("")), False)

    @abstractmethod
    def map(self, table: pa.Table) -> pa.Array:
        """Maps the given table to an output array."""

    @property
    @abstractmethod
    def all_missing(self) -> bool:
        """Indicates if all observed values were missing or empty"""


class FailOnMissingMapper(BaseMapper):
    def map(self, table: pa.Table):
        array = table.column(self._column)
        is_valid = self._is_valid(array)
        all_valid = pc.all(is_valid)
        if not all_valid.as_py():
            raise ValueError(
                f"There are missing or empty values in column {self._column}. "
                f"See row ID <{self._get_row_id_of_first_null(table, is_valid)}> "
                "for the first row that contains such a value."
            )

        results = self._fn(array.to_pylist())
        return pa.array(results, type=self._output_type)

    def _get_row_id_of_first_null(self, table, is_valid):
        empties = pc.filter(table, pc.invert(is_valid))
        return empties[0][0].as_py()

    @property
    def all_missing(self) -> bool:
        return False


class OutputMissingMapper(BaseMapper):
    def __init__(
        self,
        column: str,
        fn: Callable,
        output_type,
    ) -> None:
        super().__init__(column, fn, output_type)
        self._all_missing = True

    @property
    def all_missing(self):
        return self._all_missing

    def map(self, table: pa.Table):
        text_array = table.column(self._column)
        is_valid = self._is_valid(text_array)
        if pc.any(is_valid).as_py():
            self._all_missing = False

        all_valid = pc.all(is_valid).as_py()
        non_empty = text_array if all_valid else text_array.filter(is_valid)

        processed_values = self._fn(non_empty.to_pylist())
        processed_array = pa.array(processed_values, type=self._output_type)
        if all_valid:
            return processed_array

        # pc.replace_with_mask does not support list types for some reason
        # so we do that part ourselves here
        j = 0
        results = [None] * len(text_array)
        for k in range(len(text_array)):
            if is_valid[k].as_py():
                results[k] = processed_array[j]
                j += 1

        return pa.array(results, type=self._output_type)


async def abatched_apply(afn, inputs: list, batch_size: int) -> list:
    outputs = []
    for batch in _generate_batches(inputs, batch_size):
        outputs.extend(await afn(batch))
    return outputs


def batched_apply(fn: Callable[[list], list], inputs: list, batch_size: int) -> list:
    outputs = []
    for batch in _generate_batches(inputs, batch_size):
        outputs.extend(fn(batch))
    return outputs


def _generate_batches(inputs: list, batch_size: int):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]


class ProgressTracker:
    def __init__(self, total_rows: int, ctx: knext.ExecutionContext):
        self.total_rows = total_rows
        self.current_progress = 0
        self.ctx = ctx

    def update_progress(self, batch_size: int):
        check_canceled(self.ctx)
        self.current_progress += batch_size
        self.ctx.set_progress(self.current_progress / self.total_rows)


@dataclass
class OutputColumn:
    """Stores information on an output column of a node."""

    default_name: str
    knime_type: knext.KnimeType
    pa_type: pa.DataType

    def to_knime_column(self):
        column = knext.Column(self.knime_type, self.default_name)
        return column


def create_empty_table(
    table: knext.Table, output_columns: List[OutputColumn]
) -> knext.Table:
    """Constructs an empty KNIME Table with the correct output columns."""
    if table is None:
        pa_table = pa.table([])
    else:
        pa_table = table.to_pyarrow()

    for col in output_columns:
        output_column_name = handle_column_name_collision(
            table.schema.column_names if table is not None else [], col.default_name
        )
        pa_table = pa_table.append_column(
            output_column_name,
            pa.array([], col.pa_type),
        )
    return knext.Table.from_pyarrow(pa_table)
