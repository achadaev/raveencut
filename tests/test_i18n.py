import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


def test_set_language_ru_translates_sample_strings():
    from core.i18n import set_language

    set_language("ru")
    assert _("Export") == "Экспорт"
    assert _("Done") == "Готово"
    assert _("Browse…") == "Обзор…"


def test_set_language_en_returns_msgid():
    from core.i18n import set_language

    set_language("en")
    assert _("Export") == "Export"
    assert _("Done") == "Done"


def test_set_language_ru_format_strings():
    from core.i18n import set_language

    set_language("ru")
    assert _("Threshold: {v:.2f}").format(v=0.5) == "Порог: 0.50"
    assert _("Cutting {done}/{total}{eta}").format(done=1, total=5, eta="") == "Нарезка 1/5"


def test_unknown_key_returns_itself():
    from core.i18n import set_language

    set_language("ru")
    assert _("__unknown_key__") == "__unknown_key__"
