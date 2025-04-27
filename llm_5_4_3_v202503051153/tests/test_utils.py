import sys

sys.path.append("../src")

import unittest
from util import handle_column_name_collision

# TODO run tests during every build


class HandleColumnNameCollisionTest(unittest.TestCase):
    def test_no_collision(self):
        self.assertEqual(
            handle_column_name_collision(["column1", "column2"], "column3"), "column3"
        )

    def test_direct_collision(self):
        self.assertEqual(
            handle_column_name_collision(["column1", "column2"], "column1"),
            "column1 (#1)",
        )

    def test_incremental_collision(self):
        self.assertEqual(
            handle_column_name_collision(["column", "column (#1)"], "column"),
            "column (#2)",
        )

    def test_multiple_collision(self):
        self.assertEqual(
            handle_column_name_collision(
                ["column", "column (#1)", "column (#2)"], "column"
            ),
            "column (#3)",
        )

    def test_collision_with_mixed_numbers(self):
        self.assertEqual(
            handle_column_name_collision(["column", "column (#2)"], "column"),
            "column (#1)",
        )

    def test_collision_with_whitespace(self):
        self.assertEqual(
            handle_column_name_collision(["column", "column (#2)"], "column "),
            "column (#1)",
        )

    def test_collision_with_whitespace_and_index(self):
        self.assertEqual(
            handle_column_name_collision(["column", "column (#2)"], "column  (#1)"),
            "column  (#1)",
        )
