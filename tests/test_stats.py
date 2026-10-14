import pytest

from olmo_bench.stats import summarize


def test_summarize_known_values():
    s = summarize([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["median"] == 3
    assert s["min"] == 1 and s["max"] == 5
    assert s["iqr"] == pytest.approx(2.0)
    assert s["mean"] == 3
    assert s["cv"] == pytest.approx(1.5811388 / 3)


def test_summarize_constant_and_single():
    assert summarize([7, 7, 7])["cv"] == 0
    s = summarize([2.5])
    assert s["iqr"] == 0 and s["cv"] == 0 and s["median"] == 2.5
