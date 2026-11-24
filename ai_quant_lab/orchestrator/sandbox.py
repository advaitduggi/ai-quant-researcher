"""Strategy sandbox: execute LLM-generated code with limited surface area.

This is defense in depth, NOT a hard security boundary. Python cannot be made
fully safe against hostile code. Layers applied here:
    - Source-size cap and AST screening: import whitelist, no dunder/underscore
      attribute access (blocks the `().__class__.__base__.__subclasses__()`
      escape, including via `match` class patterns), no denylisted names, no
      frame/traceback introspection, and no numpy/pandas file/pickle/IO or
      module-internals attributes.
    - Restricted builtins (no open/eval/exec/getattr/type/__import__).
    - Execution in a *fresh interpreter* (`python -I`, new session) that does
      not inherit the parent's memory or environment (API keys never reach it),
      with CPU/file-size/core (and, on Linux, memory) rlimits and a hard
      wall-clock kill of the whole process group.
    - Pickle-free IPC: data goes in and out as plain numeric arrays (`.npz`,
      allow_pickle=False), so a compromised worker cannot attack the parent
      through deserialization.

Treat model output as untrusted: for anything beyond local research, run the
whole process in a container / VM with no secrets and add seccomp / network
isolation.
"""

from __future__ import annotations

import ast
import io
import os
import signal
import subprocess
import sys
import threading
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

MAX_SOURCE_CHARS = 100_000
_MAX_OUTPUT_BYTES = 512 * 1024 * 1024


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
    if not isinstance(price_data, (pd.Series, pd.DataFrame)):
        raise SandboxError(f"Unsupported price_data type: {type(price_data).__name__}")
    _validate_source(source)

    import time

    start = time.perf_counter()
    values = _run_isolated(source, price_data, timeout_seconds)
    elapsed = time.perf_counter() - start

    if isinstance(price_data, pd.Series):
        if values.shape != (len(price_data),):
            raise SandboxError("Strategy returned a Series with a different shape.")
        positions: pd.Series | pd.DataFrame = pd.Series(
            values, index=price_data.index, name=price_data.name
        )
    else:
        if values.shape != price_data.shape:
            raise SandboxError("Strategy returned a DataFrame with a different shape.")
        positions = pd.DataFrame(values, index=price_data.index, columns=price_data.columns)
    return SandboxResult(positions=positions.fillna(0.0), elapsed_seconds=elapsed)


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
        "mro", "os", "sys", "subprocess", "builtins", "importlib", "ctypes",
        "pickle", "shutil", "socket", "mmap", "frombuffer_", "with_traceback",
    }
)
# Frame / generator / traceback / code-object introspection, I/O readers, privates.
_FORBIDDEN_ATTR_PREFIXES: tuple[str, ...] = (
    "read_", "to_", "_", "f_", "gi_", "cr_", "ag_", "tb_", "co_", "func_",
)
# `to_` is mostly I/O (to_csv, to_pickle, to_sql, ...) but a few are harmless
# conversions that strategies legitimately use.
_ALLOWED_TO_ATTRS: frozenset[str] = frozenset(
    {"to_numpy", "to_list", "to_dict", "to_frame", "to_series", "to_datetime",
     "to_numeric", "to_timedelta", "to_period", "to_timestamp"}
)


def _check_attr_name(attr: str) -> None:
    if attr in _ALLOWED_TO_ATTRS:
        return
    if attr in _FORBIDDEN_ATTRS or attr.startswith(_FORBIDDEN_ATTR_PREFIXES):
        raise SandboxError(f"Disallowed attribute access: .{attr}")


def _validate_source(source: str) -> None:
    """Reject disallowed imports and escape-prone constructs before exec."""
    if len(source) > MAX_SOURCE_CHARS:
        raise SandboxError(f"Strategy source too large (> {MAX_SOURCE_CHARS} chars).")
    if "\x00" in source:
        raise SandboxError("Strategy source contains a NUL byte.")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error: {exc}") from exc
    except (RecursionError, MemoryError, ValueError) as exc:
        raise SandboxError(f"Strategy source could not be parsed safely: {type(exc).__name__}") from exc

    try:
        nodes = list(ast.walk(tree))
    except RecursionError as exc:
        raise SandboxError("Strategy source is nested too deeply.") from exc

    for node in nodes:
        if isinstance(node, ast.Attribute):
            _check_attr_name(node.attr)
        elif isinstance(node, ast.MatchClass):
            # `case pd.Series(__class__=c)` reads attributes without an ast.Attribute.
            for kwd in node.kwd_attrs:
                _check_attr_name(kwd)
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise SandboxError(f"Disallowed name: {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxError("global/nonlocal are not allowed.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            text = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
            if "__" in text:
                raise SandboxError("Dunder strings are not allowed.")

        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name not in _ALLOWED_IMPORTS:
                    raise SandboxError(f"Disallowed import: {name.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod not in _ALLOWED_IMPORTS:
                raise SandboxError(f"Disallowed import from: {mod}")
            for alias in node.names:
                _check_attr_name(alias.name)


def _encode_input(source: str, price_data: pd.Series | pd.DataFrame, timeout: float) -> bytes:
    """Serialize inputs as plain numeric/unicode arrays (no pickle)."""
    idx = price_data.index
    arrays: dict[str, Any] = {
        "source": np.frombuffer(source.encode("utf-8"), dtype=np.uint8),
        "values": np.asarray(price_data.to_numpy(), dtype="float64"),
    }
    if isinstance(idx, pd.DatetimeIndex):
        arrays["index_kind"] = np.array("datetime")
        naive = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
        arrays["index"] = naive.to_numpy().astype("datetime64[ns]")
        if idx.tz is not None:
            arrays["tz"] = np.array(str(idx.tz))
    elif isinstance(idx, pd.RangeIndex):
        arrays["index_kind"] = np.array("range")
        arrays["index"] = np.arange(len(idx))
    else:
        arrays["index_kind"] = np.array("other")
        try:
            arrays["index"] = np.asarray(idx.to_numpy(), dtype="float64")
        except (TypeError, ValueError):
            arrays["index"] = np.asarray([str(v) for v in idx], dtype="U")
    if isinstance(price_data, pd.DataFrame):
        arrays["columns"] = np.asarray([str(c) for c in price_data.columns], dtype="U")
    buf = io.BytesIO()
    np.savez(buf, **arrays, timeout=np.float64(timeout))
    return buf.getvalue()


def _run_isolated(source: str, price_data: Any, timeout_seconds: float) -> np.ndarray:
    """Run in a fresh interpreter; hard-kill its process group on timeout."""
    from pathlib import Path

    worker = Path(__file__).with_name("_sandbox_worker.py")
    pkg_root = str(Path(__file__).resolve().parents[2])
    payload = _encode_input(source, price_data, timeout_seconds)

    # Minimal environment: nothing from the parent (API keys, tokens) is inherited.
    env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    posix = hasattr(os, "killpg")
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-I", str(worker), pkg_root],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=os.path.dirname(str(worker)) if not posix else "/",
        start_new_session=posix,
    )

    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        try:
            if posix:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    timer = threading.Timer(timeout_seconds, _kill)
    timer.daemon = True
    timer.start()
    try:
        def _feed() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(payload)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        assert proc.stdout is not None
        data = proc.stdout.read(_MAX_OUTPUT_BYTES + 1)
    finally:
        timer.cancel()
        _kill_quiet(proc, posix)
        proc.wait()
        if proc.stdout:
            proc.stdout.close()

    if timed_out.is_set():
        raise SandboxError("Strategy exceeded the wall-clock timeout.")
    if len(data) > _MAX_OUTPUT_BYTES:
        raise SandboxError("Strategy output too large.")
    if not data:
        raise SandboxError("Strategy process died (resource limit or crash).")
    status, body = data[:1], data[1:]
    if status == b"E":
        raise SandboxError(body.decode("utf-8", "replace")[:2000])
    if status != b"O":
        raise SandboxError("Strategy process returned a malformed result.")
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            return np.asarray(z["values"], dtype="float64")
    except Exception as exc:  # noqa: BLE001
        raise SandboxError("Strategy process returned a malformed result.") from exc


def _kill_quiet(proc: subprocess.Popen, posix: bool) -> None:
    """Kill any stragglers (including grandchildren) in the worker's process group."""
    try:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
