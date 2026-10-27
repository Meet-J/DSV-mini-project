"""Small shared helpers for the generated Markdown reports."""
from __future__ import annotations

import pandas as pd


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table. `fmt` maps column -> format string, e.g. {"rate": "{:.1f}"}."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    # astype(object) stops iterrows from upcasting all-numeric rows to float (which would print 7043 as 7043.0)
    for _, row in df.astype(object).iterrows():
        cells = ["n/a" if _is_missing(row[c]) else (fmt[c].format(row[c]) if c in fmt else str(row[c])) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
