"""Where data and figures live, and the _DD_MM_YYYY naming of outputs."""
import datetime
import re
from pathlib import Path

ROOT = Path("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations")
DATA = ROOT/"analysis/output/chicago_poster_2026"
DATA_OLD = ROOT/"analysis/output"
FIGS = ROOT/"figures"
ARTICLE_FIGS = ROOT/"article_chaos/figures_article"
CODE_PYTHON = ROOT.parent/"Code_python"            # Plot_setting.py


def today():
    return datetime.date.today().strftime("%d_%m_%Y")


def dated(name, ext="json", date=None, suffix=""):
    # "name_DD_MM_YYYY<suffix>.ext", today's date by default
    return f"{name}_{date or today()}{suffix}.{ext}"


def latest(name, folder=DATA, ext="json", suffix=""):
    # newest file folder/name_DD_MM_YYYY<suffix>.ext
    pat = re.compile(rf"^{re.escape(name)}_(\d\d)_(\d\d)_(\d{{4}}){re.escape(suffix)}\.{ext}$")
    found = [(m.group(3), m.group(2), m.group(1), p) for p in Path(folder).iterdir() if (m := pat.match(p.name))]
    if not found:
        raise FileNotFoundError(f"no {name}_DD_MM_YYYY{suffix}.{ext} in {folder}")
    return max(found)[-1]
