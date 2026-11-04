"""Plain-text tables for compact CLI summaries."""

from collections.abc import Sequence


def format_fraction(value: float | None) -> str:
    """Render a 0–1 fraction as a percentage, or '-' when undefined."""
    return "-" if value is None else f"{value:.1%}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Left-align text columns and right-align numeric columns under a rule."""
    cells = [[str(value) for value in row] for row in rows]
    numeric = [
        bool(rows) and all(isinstance(row[i], (int, float)) for row in rows)
        for i in range(len(headers))
    ]
    widths = [
        max([len(header), *(len(row[i]) for row in cells)])
        for i, header in enumerate(headers)
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            value.rjust(width) if right else value.ljust(width)
            for value, width, right in zip(values, widths, numeric)
        ).rstrip()

    return "\n".join(
        [line(headers), line(["-" * width for width in widths])]
        + [line(row) for row in cells]
    )
