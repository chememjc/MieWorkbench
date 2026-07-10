# =============================================================================
# test_emit_csv_e2e.py — end-to-end --emit-csv validation: a cheap synthetic
# scene (source -> coated BK7 slab -> detector) traced through the real
# run_trace.main() + post_process.main(), asserting:
#   * NO results/<case>/data/ directory is created when --emit-csv is off
#     (byte-identical-behavior guarantee for the existing PNG/report path)
#   * every data/index.csv row names a file that actually exists
#   * the detected-spectrum CSV's total power matches report.json's
#     per-detector total_power_W
#   * every library-derived CSV (materials n/k, coating R/T) carries a
#     non-empty reference/provenance column on every row
#
# Run: /home3/optics/env/bin/python -m pytest scripts/raytracer/tests/test_emit_csv_e2e.py -q
# =============================================================================
import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                             # noqa: E402
from raytracer.tests import scenehelpers as sh            # noqa: E402


def _scene_model():
    src = sh.source_body(power_mW=5.0, coherent=False, half=5e-4,
                         lambdac_nm=550.0, x=-0.02)
    slab = sh.slab_body("Filt", "bk7", 0.000, 0.005, half=0.01,
                        coating={"Filt.Pad.Face2": "bs_5050_vis_45"})
    det = sh.detector_body(x=0.03, half=0.015)
    return sh.make_model([src, slab, det])


@pytest.fixture(scope="module")
def traced_case(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("emit_csv_case")
    model = _scene_model()
    common.validate_model(model)
    mj = tmp_path / "model.json"
    mj.write_text(json.dumps(model))
    case = tmp_path / "case"

    import run_trace
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "20000", "--nlambda", "2", "--resolution", "64",
        "--power-floor", "1e-8",
    ])
    assert rc == 0
    return mj, case


def test_no_data_dir_when_flag_off(traced_case):
    mj, case = traced_case
    import post_process
    rc = post_process.main(["--case-dir", str(case), "--model-json",
                            str(mj)])
    assert rc == 0
    assert not (case / "data").exists()


def test_emit_csv_index_spectrum_and_references(traced_case):
    mj, case = traced_case
    import post_process
    rc = post_process.main(["--case-dir", str(case), "--model-json",
                            str(mj), "--emit-csv"])
    assert rc == 0
    data = case / "data"
    assert data.exists()

    index_path = data / "index.csv"
    assert index_path.exists()
    with open(index_path) as fh:
        index_rows = list(csv.DictReader(fh))
    assert index_rows, "index.csv has no rows"
    for row in index_rows:
        assert (data / row["file"]).exists(), \
            "index.csv references missing file %r" % row["file"]

    report = json.loads((case / "report.json").read_text())
    checked_spectrum = 0
    for label, det_report in report["detectors"].items():
        safe = label.replace(".", "_")
        spec_path = data / ("spectrum_%s.csv" % safe)
        assert spec_path.exists()
        with open(spec_path) as fh:
            rows = list(csv.DictReader(fh))
        total = sum(float(r["power_W"]) for r in rows)
        assert total == pytest.approx(det_report["total_power_W"], rel=1e-9)
        checked_spectrum += 1
    assert checked_spectrum > 0

    ref_checked = 0
    for f in list(data.glob("nk_*.csv")) + list(data.glob("coating_*.csv")):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "%s has no rows" % f.name
        for row in rows:
            assert row["reference"].strip(), \
                "%s row missing reference: %r" % (f.name, row)
        ref_checked += 1
    assert ref_checked > 0, "no library-derived CSV (nk_*/coating_*) emitted"
