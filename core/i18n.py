import builtins
import gettext as _gettext
import sys
from pathlib import Path


def _locale_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "locale"
    return Path(__file__).parent.parent / "locale"


def set_language(lang: str) -> None:
    translation = _gettext.translation(
        "raveencut", localedir=_locale_dir(), languages=[lang], fallback=True
    )
    translation.install()


# Fallback no-op _ until set_language() installs the real translator
builtins._ = lambda s: s

# Default language
set_language("ru")
