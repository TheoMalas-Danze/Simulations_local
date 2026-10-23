"""Run the tests without pytest:  python3.11 pylib/tests/run.py [fast|all] [name filter]
fast (default): test_hill.py, a few seconds. all: also test_reference.py (against the saved outputs, ~2 min)."""
import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

which = sys.argv[1] if len(sys.argv) > 1 else "fast"
filt = sys.argv[2] if len(sys.argv) > 2 else ""
modules = ["test_hill"] + (["test_reference"] if which == "all" else [])
failed = 0
for name in modules:
    mod = importlib.import_module(name)
    for fname in [f for f in dir(mod) if f.startswith("test_") and filt in f]:
        t0 = time.time()
        try:
            getattr(mod, fname)()
            print(f"ok    {name}.{fname} ({time.time() - t0:.1f} s)")
        except Exception:
            failed += 1
            print(f"FAIL  {name}.{fname} ({time.time() - t0:.1f} s)")
            traceback.print_exc()
print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
