"""Regression tests for sandbox escape attempts."""

import numpy as np
import pandas as pd
import pytest

from ai_quant_lab.orchestrator.sandbox import SandboxError, run_strategy

PX = pd.Series(np.arange(30.0) + 100, index=pd.date_range("2020-01-01", periods=30))

ESCAPES = {
    "subclasses": "def strategy(p):\n    return ().__class__.__base__.__subclasses__()",
    "dunder_str": "def strategy(p):\n    return '__class__'",
    "to_csv": "def strategy(p):\n    p.to_csv('/tmp/x')\n    return p",
    "read_csv": "def strategy(p):\n    pd.read_csv('/etc/hosts')\n    return p",
    "np_load": "def strategy(p):\n    np.load('x.npy', allow_pickle=True)\n    return p",
    "np_lib": "def strategy(p):\n    np.lib.npyio\n    return p",
    "getattr": "def strategy(p):\n    return getattr(p, 'x')",
    "type": "def strategy(p):\n    return type(p)",
    "eval": "def strategy(p):\n    return eval('1')",
    "import_os": "import os\ndef strategy(p):\n    return p",
}


@pytest.mark.parametrize("name", ESCAPES)
def test_escape_blocked(name):
    with pytest.raises(SandboxError):
        run_strategy(ESCAPES[name], PX)


def test_legit_strategy_still_runs():
    src = (
        "def strategy(p):\n"
        "    ma = p.rolling(5).mean()\n"
        "    return (p > ma).astype(float).shift(1).fillna(0.0)\n"
    )
    out = run_strategy(src, PX)
    assert out.positions.index.equals(PX.index)


def test_timeout_is_hard_even_if_exception_swallowed():
    src = (
        "def strategy(p):\n"
        "    while True:\n"
        "        try:\n"
        "            while True:\n"
        "                pass\n"
        "        except BaseException:\n"
        "            pass\n"
    )
    with pytest.raises(SandboxError, match="timeout"):
        run_strategy(src, PX, timeout_seconds=1)



MORE_ESCAPES = {
    "match_class_dunder": (
        "def strategy(p):\n    match p:\n        case pd.Series(__class__=c):\n            return c\n    return p"
    ),
    "from_import_reader": "from pandas import read_pickle\ndef strategy(p):\n    return p",
    "import_submodule": "import numpy.ctypeslib\ndef strategy(p):\n    return p",
    "module_os_attr": "def strategy(p):\n    return np.os",
    "frame_walk": "def strategy(p):\n    g = (x for x in [1])\n    return g.gi_frame",
    "bytes_dunder": "def strategy(p):\n    return b'__class__'",
    "huge_source": "x = 1\n" * 100_000,
    "deep_nesting": "def strategy(p):\n    return " + "(" * 5000 + "p" + ")" * 5000,
}


@pytest.mark.parametrize("name", MORE_ESCAPES)
def test_more_escapes_blocked(name):
    with pytest.raises(SandboxError):
        run_strategy(MORE_ESCAPES[name], PX)


def test_worker_does_not_inherit_secrets(monkeypatch):
    """The worker is launched with a minimal env: no API keys or other parent vars."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("MY_PRIVATE_VALUE", "leak-me")
    from ai_quant_lab.orchestrator import sandbox

    captured = {}
    real_popen = sandbox.subprocess.Popen

    def spy(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(sandbox.subprocess, "Popen", spy)
    run_strategy("def strategy(p):\n    return p * 0.0", PX)
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "MY_PRIVATE_VALUE" not in captured["env"]


def test_result_is_pickle_free_numeric_only():
    """A strategy returning non-numeric junk is rejected, not unpickled."""
    src = "def strategy(p):\n    return pd.Series(['a'] * len(p), index=p.index)"
    with pytest.raises(SandboxError):
        run_strategy(src, PX)


def test_roundtrip_preserves_index_and_tz():
    idx = pd.date_range("2020-01-01", periods=20, tz="America/New_York")
    px = pd.Series(np.arange(20.0) + 100, index=idx, name="close")
    out = run_strategy("def strategy(p):\n    return (p > p.mean()).astype(float)", px)
    assert out.positions.index.equals(idx)
    assert out.positions.name == "close"


def test_settings_does_not_leak_api_key():
    from ai_quant_lab.config import Settings

    s = Settings(anthropic_api_key="sk-ant-secret-123")
    assert "sk-ant-secret-123" not in repr(s)
    assert "sk-ant-secret-123" not in str(s.model_dump())
    assert s.require_api_key() == "sk-ant-secret-123"
