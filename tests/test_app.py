"""Smoke tests: both views render without errors and show the key answers."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from gravity_inverted.data import has_wiod  # noqa: E402

pytestmark = pytest.mark.skipif(not has_wiod(), reason="WIOD data not built")
APP = str(Path(__file__).parents[1] / "app.py")


def run(**params):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in params.items():
        at.query_params[k] = v
    return at.run()


def text(at):
    return " ".join(m.value for m in at.markdown)


def test_plain_view_defaults():
    at = run()
    assert not at.exception and not at.error
    t = text(at)
    assert "Germany" in t and "times as large" in t and "the United States" in t
    assert "1.1 %" in t  # domestic instrument for a 1 % gain (Table D1: -1.13)


def test_plain_view_reads_shared_link():
    at = run(c="FRA", g="2.0", w="CHN", s="30", a="DEU", b="FRA", y="2010")
    assert not at.exception and not at.error
    t = text(at)
    assert "France" in t and "2 % richer" in t and "30 % more expensive" in t and "2010" in t


def test_plain_view_ignores_bad_link():
    at = run(c="XXX", g="99", s="abc")
    assert not at.exception and not at.error
    assert "Germany" in text(at)


def test_expert_view():
    at = run(view="expert")
    assert not at.exception and not at.error
    assert any("DEU" in m.label for m in at.metric)
