"""Plot style of the repository: `from catchaos.plotting import *` re-exports Code_python/Plot_setting.py
(figsize, figsize_car, Perso_palette, Perso_palette_float, Perso_colormap, ...)."""
import sys

from .paths import CODE_PYTHON

if str(CODE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CODE_PYTHON))

import Plot_setting                       # noqa: E402  (sets the matplotlib rcParams)
from Plot_setting import *                # noqa: E402,F401,F403
