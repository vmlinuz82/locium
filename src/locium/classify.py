"""Content-shape classification: prose vs data dump vs tool output.

A mined palace stores whatever crossed the transcript: alongside real
reasoning sit pasted CSVs, `ls -la` listings, grep dumps and diff hunks.
Those embed strongly and rank as memories while carrying no reusable
knowledge, so the viewer needs to know which is which -- to collapse them in
search results and to report a noise share on the health line. Classification
is display-side only; nothing is ever excluded from the index.

Two complications shape the heuristics. Exchange-mode chunks had their
newlines collapsed to spaces at mine time, so line-based rules alone are
blind to a single-line CSV -- hence the density signals over the whole text.
And prose that merely QUOTES a command or one CSV row must stay prose --
hence thresholds on shares, never on presence.
"""

import re

_TOOL_CALL_RE = re.compile(
    r"\[(?:Bash|Read|Grep|Glob|Edit|Write|MultiEdit|NotebookEdit|LS|Task|"
    r"TodoWrite|WebFetch|WebSearch|KillShell|BashOutput|mcp__\w+)\b[^\]]*\]"
)
_PERMISSION_RE = re.compile(r"[-ld][rwxst-]{9}\s+\d+\s+\S+")
_DIFF_LINE_RE = re.compile(r"^(?:diff --git |index [0-9a-f]{7,}\.\.|@@ |\+\+\+ |--- |[+-][^+-])")

# One quoted CSV field boundary. '","' is the signature of quoted CSV rows
# and essentially never occurs in prose.
_CSV_SEAM = '","'


def classify(text: str) -> str | None:
    """The drawer's content shape: "noise", "data", or None for prose.

    "noise" is tool traffic (results, call markers, listings); "data" is
    structured payload (CSV, diffs). Prose returns None so callers can store
    the field only when it says something.
    """
    stripped = text.strip()
    if not stripped:
        return None

    lines = [line for line in stripped.split("\n") if line.strip()]

    tool_lines = sum(
        1 for line in lines
        if line.strip().startswith("→") or _TOOL_CALL_RE.match(line.strip())
    )
    diff_lines = sum(1 for line in lines if _DIFF_LINE_RE.match(line.strip()))

    # Density signals survive the miner's newline-collapsing.
    tool_marks = len(_TOOL_CALL_RE.findall(stripped))
    permission_marks = len(_PERMISSION_RE.findall(stripped))
    csv_seams = stripped.count(_CSV_SEAM)

    if tool_lines / len(lines) >= 0.4:
        return "noise"
    # Roughly one tool/permission mark per 120 chars means the text IS the
    # tool traffic, not prose that mentions it.
    if (tool_marks + permission_marks) * 120 >= len(stripped) and (
        tool_marks + permission_marks
    ) >= 3:
        return "noise"

    if len(lines) >= 4 and diff_lines / len(lines) >= 0.5:
        return "data"
    # A CSV seam every ~150 chars, at least three of them: a data row dump.
    if csv_seams >= 3 and csv_seams * 150 >= len(stripped):
        return "data"

    return None
