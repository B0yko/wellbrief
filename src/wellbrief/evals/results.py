"""Metadata and JSON output of an evaluation run.

A result file says where its numbers came from: the UTC date, the hardware
(CPU model, RAM, operating system), the Python version, the git commit and
whether the tree was dirty, the seeds, the wellbrief version, the suites and
the command line. It never records the machine's name: no field comes from
`platform.node()`, `socket.gethostname()` or the node name of `uname`. Paths
in the command line are written relative to the working directory, and any
absolute path that ends up in a case detail (an exception message, say) is
rewritten before the file is written.

The dirty flag ignores untracked files under `results/`: a result file written
by an earlier run of the same session is output, not a change to the code the
numbers come from.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHECKOUT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
_RESULTS_UNTRACKED = "?? results/"
# Absolute paths left after the known prefixes are replaced: any /a/b... token, or a
# single-segment path under a well-known root, and relative paths that climb out
# (../x). A hole size such as 12 1/4" never matches: its slash follows a digit.
_TAIL = r"[^\s'\",;:)\]]*"
_ABSOLUTE = re.compile(
    rf"(?<![\w.~<>/-])/(?:[A-Za-z_][\w.@+-]*/{_TAIL}"
    rf"|(?:Users|home|private|var|tmp|opt|root|mnt|Volumes|usr|app|srv|etc|github|workspace|builds|nix|run)\b{_TAIL})"
    rf"|(?<![\w.~<>/-])(?:\.\./)+{_TAIL}")


def _run(*cmd: str) -> str | None:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def cpu_model() -> str:
    if platform.system() == "Darwin":
        model = _run("sysctl", "-n", "machdep.cpu.brand_string")
        if model:
            return model
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"model name", "Model", "Hardware"} and value.strip():
                return value.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    if platform.system() == "Darwin":
        size = _run("sysctl", "-n", "hw.memsize")
        if size and size.isdigit():
            return round(int(size) / 2**30, 1)
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) * 1024 / 2**30, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}".strip()
    if system == "Linux":
        try:
            return platform.freedesktop_os_release().get("PRETTY_NAME", "Linux")
        except OSError:
            return f"Linux {platform.release()}"
    return f"{system} {platform.release()}".strip()


def git_state(root: Path = _CHECKOUT) -> dict[str, Any]:
    """Commit and dirty flag of the source checkout wellbrief runs from ("unknown" outside one)."""
    if not (root / ".git").exists() or not (root / "src" / "wellbrief").is_dir():
        return {"sha": "unknown", "dirty": "unknown"}
    sha = _run("git", "-C", str(root), "rev-parse", "HEAD")
    status = _run("git", "-C", str(root), "status", "--porcelain")
    if sha is None or status is None:
        return {"sha": "unknown", "dirty": "unknown"}
    changes = [line for line in status.splitlines() if line and not line.startswith(_RESULTS_UNTRACKED)]
    return {"sha": sha, "dirty": bool(changes)}


def _relative(arg: str) -> str:
    """A path inside the working directory becomes relative; any other absolute path keeps only its name."""
    if arg.startswith("--") and "=" in arg:
        flag, _, value = arg.partition("=")
        return f"{flag}={_relative(value)}"
    if not os.path.isabs(arg):
        return arg
    path, cwd = Path(os.path.realpath(arg)), Path(os.path.realpath(os.getcwd()))
    if path == cwd or cwd in path.parents:
        return path.relative_to(cwd).as_posix() or "."
    return f"<outside>/{path.name}"


def command_line(args: list[str]) -> str:
    """The eval command as typed, with every path relative to the working directory."""
    return " ".join(["wellbrief", *(_relative(a) for a in args)])


def metadata(args: list[str], suites: list[str], seeds: list[int], options: dict[str, Any],
             version: str) -> dict[str, Any]:
    return {
        "wellbrief_version": version,
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command_line(args),
        "suites": suites,
        "seeds": seeds,
        "options": options,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "hardware": {"cpu": cpu_model(), "ram_gb": ram_gb(), "os": os_name(), "arch": platform.machine()},
        "git": git_state(),
    }


def _replacements() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for base, label in ((tempfile.gettempdir(), "<tmp>"), (os.getcwd(), "."), (str(Path.home()), "~"),
                        (sys.prefix, "<python>"), (sys.base_prefix, "<python>"), (str(_PACKAGE), "<wellbrief>")):
        for form in {base, os.path.realpath(base)}:
            if len(form) > 1:
                pairs.append((form, label))
    return sorted(pairs, key=lambda p: -len(p[0]))


def scrub(obj: Any, replacements: list[tuple[str, str]] | None = None) -> Any:
    """Rewrite absolute paths in every string of a JSON-ready structure."""
    replacements = _replacements() if replacements is None else replacements
    if isinstance(obj, str):
        for old, new in replacements:
            obj = obj.replace(old, new)
        return _ABSOLUTE.sub("<path>", obj)
    if isinstance(obj, dict):
        return {k: scrub(v, replacements) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, replacements) for v in obj]
    return obj


def write(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scrub(report), indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
