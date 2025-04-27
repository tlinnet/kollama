import knime.extension as knext
from typing import List
from dataclasses import dataclass
import pyarrow as pa
import re


ai_icon = "icons/ml.png"
main_category = knext.category(
    path="/community",
    level_id="kollama",
    name="Kollama",
    description="",
    icon="icons/kollama.png",
)


@dataclass
class OutputColumn:
    """Stores information on an output column of a node."""

    default_name: str
    knime_type: knext.KnimeType
    pa_type: pa.DataType

    def to_knime_column(self):
        column = knext.Column(self.knime_type, self.default_name)
        return column


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