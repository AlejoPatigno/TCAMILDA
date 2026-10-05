"""Run code in a separate Python interpreter that never imports TensorFlow.

openSMILE and PyTorch crash (heap corruption / segfault) when they run in a
process where TensorFlow is already loaded, which is the case in the
experiment notebook. Feature extraction and PyTorch training therefore run in a
fresh interpreter. A plain subprocess is used instead of multiprocessing
"spawn", which would re-import the caller's main script (and TensorFlow with
it) in the child.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

_CHILD = ("import pickle, sys; fn, args = pickle.load(open(sys.argv[1], 'rb')); "
          "pickle.dump(fn(*args), open(sys.argv[2], 'wb'))")


def run_isolated(fn, *args, stream_output: bool = False):
    """Run ``fn(*args)`` (a module-level function) in a separate interpreter and return its result.

    With ``stream_output`` the child's stdout/stderr go to the caller's console
    (progress of long trainings); otherwise they are captured and shown only on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        with open(inp, "wb") as fh:
            pickle.dump((fn, args), fh)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        cmd = [sys.executable, "-c", _CHILD, str(inp), str(out)]
        if stream_output and _has_fileno():
            proc = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
            detail = ""
        elif stream_output:  # Jupyter: no file descriptor to share, so print the log afterwards
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(proc.stdout, end="")
            detail = proc.stderr[-3000:]
        else:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            detail = proc.stderr[-3000:]
        if proc.returncode != 0:
            raise RuntimeError(f"Isolated run of {fn.__name__} failed (exit {proc.returncode}):\n{detail}")
        with open(out, "rb") as fh:
            return pickle.load(fh)


def _has_fileno() -> bool:
    try:
        sys.stdout.fileno()
        sys.stderr.fileno()
        return True
    except Exception:  # Jupyter streams have no file descriptor
        return False
