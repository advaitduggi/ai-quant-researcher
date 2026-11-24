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

