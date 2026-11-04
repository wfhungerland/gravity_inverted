"""Smoke tests: both views render without errors."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from gravity_inverted.data import has_wiod  # noqa: E402

pytestmark = pytest.mark.skipif(not has_wiod(), reason="WIOD data not built")


def run(expert: bool):
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120).run()
    if expert:
        at.sidebar.toggle[0].set_value(True).run()
    return at


def test_simple_view():
    at = run(expert=False)
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "Germany" in text and "times bigger" in text


def test_expert_view():
    at = run(expert=True)
    assert not at.exception and not at.error
    assert any("DEU" in m.label for m in at.metric)
