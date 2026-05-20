"""Compact formatting utilities for numerical logging output.

Provides consistent number/array/table formatting without heavy dependencies.
Designed for log messages and console output with controlled precision and layout.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union
import textwrap
import numpy as np

Number = Union[int, float, np.floating, np.integer]


def fmt_num(x: Number, digits: int = 3, sci: bool | None = None, sign: bool = False) -> str:
    """Format number with controlled precision and optional scientific notation.

    Parameters
    ----------
    x : number
        Value to format (int, float, numpy scalar)
    digits : int
        Decimal places (fixed) or significant figures (scientific)
    sci : bool | None
        Force scientific (True), fixed (False), or auto-detect (None)
        Auto uses scientific if |x| < 1e-3 or |x| >= 1e4
    sign : bool
        Always show sign prefix (+/-)
        
    Returns
    -------
    str
        Formatted number string, e.g., "1.234e-02" or "+3.142"
    """
    if not np.isfinite(x):
        return str(x)
    if sci is None:
        ax = abs(float(x))
        sci = ax != 0 and (ax < 1e-3 or ax >= 1e4)
    if sci:
        fmt = f"%{'+' if sign else ''}.{digits}e"
        return fmt % float(x)
    fmt = f"%{'+' if sign else ''}.{digits}f"
    return fmt % float(x)


def fmt_shape(a: np.ndarray | Sequence[int] | Tuple[int, ...]) -> str:
    """Format array shape as compact tuple string, e.g., '(128, 64, 3)'."""
    if isinstance(a, np.ndarray):
        shp = a.shape
    else:
        shp = tuple(int(v) for v in a)
    return f"{tuple(shp)}"


def fmt_list(items: Iterable, max_items: int = 12) -> str:
    """Format iterable as single-line list with truncation.
    
    Shows first max_items elements, appends '... +N' if truncated.
    Numbers formatted via fmt_num() for consistency.
    """
    it = list(items)
    n = len(it)
    if n == 0:
        return "[]"
    shown = it[: max_items]
    parts: List[str] = []
    for v in shown:
        if isinstance(v, (int, float, np.floating, np.integer)):
            parts.append(fmt_num(float(v)))
        else:
            parts.append(str(v))
    if n > max_items:
        parts.append(f"… +{n - max_items}")
    return "[" + ", ".join(parts) + "]"


def fmt_array(a: np.ndarray, digits: int = 3, sci: bool | None = True, max_width: int = 160) -> str:
    """Format numpy array for log output with controlled layout.
    
    Uses numpy's array2string with custom formatters. Wraps wide output to max_width.
    """
    if a is None or getattr(a, "size", 0) == 0:
        return "(empty)"
    # Decide scientific formatting behavior
    if sci is True:
        float_kind = lambda x: f"{float(x): .{digits}e}"  # noqa: E731
    elif sci is False:
        float_kind = lambda x: f"{float(x): .{digits}f}"  # noqa: E731
    else:
        def float_kind(x: float) -> str:  # type: ignore[no-redef]
            return fmt_num(float(x), digits=digits, sci=None)
    return np.array2string(a, formatter={"float_kind": float_kind}, max_line_width=max_width)


def fmt_matrix_with_columns(a: np.ndarray, labels: Sequence[str], digits: int = 3,
                            sci: bool | None = True, max_width: int = 160) -> str:
    """Format a 2D array with a header listing column labels."""
    header = "columns: " + ", ".join(map(str, labels))
    body = fmt_array(a, digits=digits, sci=sci, max_width=max_width)
    return f"{header}\n{body}"


def tabulate(rows: Sequence[Sequence[Union[str, Number]]], headers: Sequence[str] | None = None,
             num_digits: int = 3, right_align_cols: Sequence[int] | None = None) -> str:
    """Minimal, dependency-free table formatter for logs.

    - Aligns columns based on max width
    - Formats numbers uniformly
    """
    def to_text(v: Union[str, Number]) -> str:
        if isinstance(v, (int, float, np.floating, np.integer)):
            return fmt_num(float(v), digits=num_digits, sci=None)
        return str(v)

    rows_txt = [[to_text(v) for v in row] for row in rows]
    if headers is not None:
        header_txt = [str(h) for h in headers]
        widths = [max(len(h), max((len(row[i]) for row in rows_txt), default=0)) for i, h in enumerate(header_txt)]
    else:
        widths = [max((len(row[i]) for row in rows_txt), default=0) for i in range(len(rows_txt[0]) if rows_txt else 0)]
        header_txt = None

    ra = set(right_align_cols or [])

    def pad_row(cols: Sequence[str]) -> str:
        parts = []
        for i, (c, w) in enumerate(zip(cols, widths)):
            if i in ra:
                parts.append(c.rjust(w))
            else:
                parts.append(c.ljust(w))
        return "  ".join(parts)

    lines: List[str] = []
    if header_txt is not None:
        lines.append(pad_row(header_txt))
        lines.append(pad_row(["-" * len(h) for h in header_txt]))
    for r in rows_txt:
        lines.append(pad_row(r))
    return "\n".join(lines)


def fmt_list_wrapped(items: Iterable, max_items: int = 20, width: int = 120,
                     indent: int = 2, bullet: bool = False) -> str:
    """Format items over multiple lines for logs.

    - If bullet=True, returns a bullet list indented by `indent` spaces.
    - Else, wraps a comma-separated line with `textwrap.fill` and applies `subsequent_indent`.
    - Returns a string that starts with a leading newline to place nicely after a preceding message.
    """
    it = list(items)
    if not it:
        return "\n" + (" " * indent) + "(none)"

    shown = it[:max_items]
    if bullet:
        lines = [(" " * indent) + "- " + str(v) for v in shown]
        if len(it) > max_items:
            lines.append((" " * indent) + f"… +{len(it) - max_items} more")
        return "\n" + "\n".join(lines)
    else:
        text = ", ".join(str(v) for v in shown)
        if len(it) > max_items:
            text += f", … +{len(it) - max_items}"
        return "\n" + textwrap.fill(text, width=width, subsequent_indent=(" " * indent))
