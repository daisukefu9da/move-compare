import shutil

import pytest

from dance_compare.validation import (Expectation, _count_pos, evaluate, load_cases, make_synthetic_set,
                                      validate)


def test_count_pos():
    assert _count_pos("1-1") == 0
    assert _count_pos("2-3") == 10
    assert _count_pos("2-3&") == 10.5


def test_example_cases_file_parses():
    settings, cases = load_cases("validation/cases.example.toml")
    assert [c.name for c in cases] == ["elbow_bent", "half_beat_late", "landing_late", "small_amplitude"]
    assert cases[0].expect[0].part == "right_elbow"


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[case]]\nname="x"\nref="a"\nuser="b"\n  [[case.expect]]\n  aspect="shape"\n  prt="x"\n')
    with pytest.raises(ValueError):
        load_cases(str(p))


def test_evaluate_reports_wrong_place_and_value():
    from dance_compare.core import compare
    from dance_compare.synth import synth_beats, synth_dancer
    from dance_compare.validation import Case

    beats = synth_beats(16)
    r = compare(synth_dancer(), synth_dancer(elbow_bend=(4.5, 6.5, 50)), ref_beats=beats, user_beats=beats,
                synced=True, first_one=0.5)
    base = dict(aspect="shape", part="right_elbow", direction="smaller")
    case = Case("x", {}, [Expectation(**base, start="2-1", end="2-4"),
                          Expectation(**base, start="4-1", end="4-4"),
                          Expectation(**base, start="2-1", end="2-4", max=-80)])
    res = evaluate(case, r, tol=1)
    assert [c.status for c in res.checks] == ["OK", "場所違い", "値が範囲外"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_synthetic_validation_set_passes(tmp_path):
    path = make_synthetic_set(str(tmp_path / "set"))
    results = validate(path, str(tmp_path / "out"), log=lambda *a: None)
    failed = [r.case.name for r in results if not r.passed]
    assert not failed, (tmp_path / "out" / "validation.md").read_text()
    assert (tmp_path / "out" / "landing_late" / "findings.csv").exists()
