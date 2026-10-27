import numpy as np
import pandas as pd

from src.reporting import md_table


def test_all_numeric_integer_columns_are_not_printed_as_floats():
    df = pd.DataFrame({"risk_decile": [1, 2], "customers": [705, 704], "rate": [75.2, 57.3]})
    out = md_table(df)
    assert "| 1 | 705 | 75.2 |" in out and "705.0" not in out


def test_missing_values_are_shown_as_n_a_and_formats_are_applied():
    df = pd.DataFrame({"band": ["a", "b"], "rate": [12.3456, np.nan]})
    out = md_table(df, {"rate": "{:.1f}"})
    assert "| a | 12.3 |" in out and "| b | n/a |" in out and "nan" not in out
