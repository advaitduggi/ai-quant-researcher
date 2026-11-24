"""Strategy sandbox: execute LLM-generated code with limited surface area.

This is defense in depth, NOT a hard security boundary. In-process Python
cannot be made fully safe against hostile code. Layers applied here:
    - Import whitelist.
    - AST screening: no dunder/underscore attribute access (blocks the
      `().__class__.__base__.__subclasses__()` escape), no denylisted names,
      and no numpy/pandas file/pickle/IO or module-internals attributes.
    - Restricted builtins (no open/eval/exec/getattr/type/__import__).
    - Execution in a forked child process: hard-killed on timeout, API keys
      scrubbed from its environment, CPU/file-size/core (and, on Linux,
      memory) rlimits applied.

Treat model output as untrusted: for anything beyond local research, run the
whole process in a container / VM with no secrets (ANTHROPIC_API_KEY is in
this process's environment) or move execution to a subprocess with
resource.setrlimit + seccomp.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class SandboxError(RuntimeError):
    """Raised when sandboxed code fails for any reason."""


@dataclass(frozen=True)
class SandboxResult:
    positions: pd.Series | pd.DataFrame
    elapsed_seconds: float


_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "numpy",
        "np",
        "pandas",
        "pd",
        "math",
        "ai_quant_lab.features.library",
        "ai_quant_lab.features.cross_sectional",
    }
)


def run_strategy(
    source: str,
    price_data: pd.Series | pd.DataFrame,
    *,
    timeout_seconds: float = 10.0,
) -> SandboxResult:
    """Execute a strategy source string against price data.

    The source must define a function `strategy(price_data)`. Single-asset
    strategies accept a Series; cross-sectional strategies accept a DataFrame.
    The engine inspects the input type and validates the output shape matches.

    Args:
        source: Python source code defining `strategy`.
        price_data: Series (single asset) or DataFrame (multi-asset).
        timeout_seconds: Wall-clock cap; the child process is killed on expiry.

    Returns:
        SandboxResult with the positions (Series or DataFrame matching input).

    Raises:
        SandboxError: on disallowed imports, missing function, exceptions,
            shape mismatch, or timeout.
    """
    _validate_source(source)

    import time

    start = time.perf_counter()
    positions = _run_isolated(source, price_data, timeout_seconds)
    elapsed = time.perf_counter() - start

    expected_type = type(price_data).__name__
    if isinstance(price_data, pd.Series):
        if not isinstance(positions, pd.Series):
            raise SandboxError(
                f"Single-asset strategy must return a Series, got {type(positions).__name__}."
            )
        if not positions.index.equals(price_data.index):
            raise SandboxError("Strategy returned a Series with a different index.")
        return SandboxResult(positions=positions.fillna(0.0), elapsed_seconds=elapsed)

    if isinstance(price_data, pd.DataFrame):
        if not isinstance(positions, pd.DataFrame):
            raise SandboxError(
                f"Cross-sectional strategy must return a DataFrame, got {type(positions).__name__}."
            )
        if not positions.index.equals(price_data.index):
            raise SandboxError("Strategy returned a DataFrame with a different index.")
        if not positions.columns.equals(price_data.columns):
            raise SandboxError("Strategy returned a DataFrame with different columns.")
        return SandboxResult(positions=positions.fillna(0.0), elapsed_seconds=elapsed)

    raise SandboxError(f"Unsupported price_data type: {expected_type}")


_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
        "eval", "exec", "compile", "open", "input", "breakpoint", "help",
        "__import__", "__builtins__", "memoryview", "super", "type",
        "os", "sys", "subprocess", "builtins", "importlib", "ctypes",
    }
)

# Attributes that give file/pickle/network access or reach module internals
# of numpy/pandas (which expose `os`, `subprocess`, `ctypes`, etc.).
_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
        "genfromtxt", "fromfile", "tofile", "fromregex", "memmap", "DataSource",
        "lib", "core", "ctypeslib", "f2py", "testing", "distutils", "compat",
        "util", "io", "api", "errors", "plotting", "show_versions", "eval",
        "query", "system", "popen", "run", "Popen", "format", "format_map",
        "mro", "gi_frame", "gi_code", "f_globals", "f_builtins", "func_globals",
    }
)
_FORBIDDEN_ATTR_PREFIXES: tuple[str, ...] = ("read_", "to_", "_")
# `to_` is mostly I/O (to_csv, to_pickle, to_sql, ...) but a few are harmless
# conversions that strategies legitimately use.
_ALLOWED_TO_ATTRS: frozenset[str] = frozenset(
    {"to_numpy", "to_list", "to_dict", "to_frame", "to_series", "to_datetime",
     "to_numeric", "to_timedelta", "to_period", "to_timestamp"}
)


def _validate_source(source: str) -> None:
    """Reject disallowed imports and escape-prone constructs before exec."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _ALLOWED_TO_ATTRS:
                continue
            if attr in _FORBIDDEN_ATTRS or attr.startswith(_FORBIDDEN_ATTR_PREFIXES):
                raise SandboxError(f"Disallowed attribute access: .{attr}")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise SandboxError(f"Disallowed name: {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxError("global/nonlocal are not allowed.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise SandboxError("Dunder strings are not allowed.")

        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise SandboxError(f"Disallowed import: {name.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in _ALLOWED_IMPORTS:
                raise SandboxError(f"Disallowed import from: {mod}")


def _safe_builtins() -> dict[str, Any]:
    """Whitelist of builtins the strategy can call.

    We allow the usual math primitives but drop `open`, `eval`, `exec`, `__import__`.
    Numpy and pandas already give the strategy everything it needs.
    """
    import builtins

    safe = {
        name: getattr(builtins, name)
        for name in (
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "len", "list", "map", "max", "min", "range", "round", "set",
            "slice", "sorted", "str", "sum", "tuple", "zip", "isinstance",
            "True", "False", "None",
        )
        if hasattr(builtins, name)
    }
    return safe


def _child_main(conn: Any, source: str, price_data: Any, timeout_seconds: float) -> None:
    """Runs in a forked child: drop secrets, apply rlimits, exec, send result."""
    import os

    try:
        # The child never needs credentials; scrub them so an escape can't read them.
        for key in list(os.environ):
            if any(m in key.upper() for m in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                del os.environ[key]
        try:
            import resource

            cpu = max(1, int(timeout_seconds) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if sys.platform.startswith("linux"):
                mem = 2 * 1024**3
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ImportError, ValueError, OSError):
            pass

        namespace: dict[str, Any] = {"np": np, "pd": pd, "__builtins__": _safe_builtins()}
        try:
            exec(compile(source, "<strategy>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            conn.send(("err", f"Strategy failed to compile: {exc}"))
            return
        fn = namespace.get("strategy")
        if not callable(fn):
            conn.send(("err", "Strategy source did not define a callable `strategy`."))
            return
        try:
            conn.send(("ok", fn(price_data)))
        except Exception as exc:
            conn.send(("err", f"Strategy raised: {exc}"))
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(("err", f"Strategy failed: {type(exc).__name__}"))
        except Exception:
            pass
    finally:
        conn.close()


def _run_isolated(source: str, price_data: Any, timeout_seconds: float) -> Any:
    """Execute in a child process that the parent can hard-kill on timeout."""
    import multiprocessing as mp

    if not hasattr(os, "fork"):
        raise SandboxError("Isolated strategy execution requires a POSIX platform.")
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_main, args=(child_conn, source, price_data, timeout_seconds), daemon=True
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_seconds):
            raise SandboxError("Strategy exceeded the wall-clock timeout.")
        try:
            status, payload = parent_conn.recv()
        except (EOFError, OSError) as exc:
            raise SandboxError("Strategy process died (resource limit or crash).") from exc
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        parent_conn.close()
    if status != "ok":
        raise SandboxError(str(payload))
    return payload
