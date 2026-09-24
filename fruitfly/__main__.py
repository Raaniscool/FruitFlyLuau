"""``python -m fruitfly`` entry point (the console script does the same).

Kept as a two-line delegation so the module path works from a bare clone without
installing the package, which is how the data-inspection workflow in DATA.md is
documented.
"""

from .cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
