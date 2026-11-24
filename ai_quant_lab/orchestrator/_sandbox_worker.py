"""Sandbox worker: runs in a fresh, isolated interpreter (``python -I``).

Protocol (no pickle anywhere, so a compromised worker cannot attack the parent
through deserialization):

    stdin : an ``.npz`` archive (allow_pickle=False) with arrays
            ``source`` (uint8 utf-8), ``values`` (float64), ``timeout`` (float64),
            ``index_kind`` / ``index`` and ``columns`` (optional, '<U' strings).
    stdout: ``b"O"`` + ``.npz`` with ``values`` (float64) on success, or
            ``b"E"`` + utf-8 error message.

The worker starts with a scrubbed environment (the parent passes none of its
secrets), applies resource limits, and has no access to the parent's memory.
Security screening of the source happens in the parent *before* launch; this
file only executes already-screened code.
"""

from __future__ import annotations

import io
import os
import sys


def _rebuild_index(kind: str, arr, tz: str):
    import pandas as pd

    if kind == "datetime":
        idx = pd.DatetimeIndex(arr.astype("datetime64[ns]"))
        return idx.tz_localize("UTC").tz_convert(tz) if tz else idx
    if kind == "range":
        return pd.RangeIndex(len(arr))
    return pd.Index(arr)


def _safe_builtins() -> dict:
    """Whitelist of builtins the strategy can call (no open/eval/exec/__import__)."""
    import builtins

    names = (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "map", "max", "min", "range", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip", "isinstance",
        "True", "False", "None",
    )
    return {n: getattr(builtins, n) for n in names if hasattr(builtins, n)}


def _apply_rlimits(timeout_seconds: float) -> None:
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


def main() -> None:
    # Keep the real stdout for the result; point fd 1 / sys.stdout at devnull so
    # nothing the strategy (or a library) prints can corrupt the protocol.
    out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115

    def fail(message: str) -> None:
        out.write(b"E" + message.encode("utf-8", "replace")[:2000])
        out.flush()

    try:
        import numpy as np
        import pandas as pd

        if len(sys.argv) > 1:
            sys.path.insert(0, sys.argv[1])
        raw = sys.stdin.buffer.read()
        data = np.load(io.BytesIO(raw), allow_pickle=False)
        source = bytes(data["source"]).decode("utf-8")
        values = data["values"]
        timeout = float(data["timeout"])
        tz = str(data["tz"]) if "tz" in data else ""
        index = _rebuild_index(str(data["index_kind"]), data["index"], tz)
        if values.ndim == 1:
            price = pd.Series(values, index=index)
        else:
            price = pd.DataFrame(values, index=index, columns=list(data["columns"]))

        _apply_rlimits(timeout)

        namespace = {"np": np, "pd": pd, "__builtins__": _safe_builtins()}
        try:
            exec(compile(source, "<strategy>", "exec"), namespace)  # noqa: S102  # nosec B102
        except Exception as exc:
            return fail(f"Strategy failed to compile: {exc}")
        fn = namespace.get("strategy")
        if not callable(fn):
            return fail("Strategy source did not define a callable `strategy`.")
        try:
            result = fn(price)
        except Exception as exc:
            return fail(f"Strategy raised: {exc}")

        if isinstance(price, pd.Series):
            if not isinstance(result, pd.Series):
                return fail(
                    f"Single-asset strategy must return a Series, got {type(result).__name__}."
                )
            if not result.index.equals(price.index):
                return fail("Strategy returned a Series with a different index.")
        else:
            if not isinstance(result, pd.DataFrame):
                return fail(
                    f"Cross-sectional strategy must return a DataFrame, got {type(result).__name__}."
                )
            if not result.index.equals(price.index):
                return fail("Strategy returned a DataFrame with a different index.")
            if not result.columns.equals(price.columns):
                return fail("Strategy returned a DataFrame with different columns.")
        try:
            arr = result.to_numpy(dtype="float64")
        except Exception as exc:
            return fail(f"Strategy output is not numeric: {type(exc).__name__}")
        buf = io.BytesIO()
        np.savez(buf, values=arr)
        out.write(b"O" + buf.getvalue())
        out.flush()
    except BaseException as exc:  # noqa: BLE001
        try:
            fail(f"Strategy failed: {type(exc).__name__}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
