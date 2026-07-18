"""Gallery optimize/tolerance-config contract (no FreeCAD, no trace).

Every committed demo ships an OptimizePane/TolerancePane config on its
miewb_vars sheet (scripts/make_demos.py DEMO_STUDIES + auto_tolerances).
This test reads each shipped .FCStd's stored config straight from the
Document.xml (stdlib zip) and asserts:

  * it round-trips through the real pane -- apply_config() drops NO row
    (a spec the pane cannot parse would silently vanish), and config()
    reproduces the same variable/operand/tolerance counts;
  * the four SHOWCASE demos ship an optimize config;
  * add()-only demos (no chained element) ship NO tolerance config.

It guards the make_demos <-> panes contract so a future demo edit that
emits a malformed spec fails loudly here instead of shipping a broken
gallery. Run under the GUI venv:

  QT_QPA_PLATFORM=offscreen env/bin/python -m pytest \
      mieworkbench/tests/test_demo_studies.py -q
"""

import html
import json
import re
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEMOS_DIR = REPO / "demos"

SHOWCASE = ["camera_triplet", "schmidt_cassegrain", "double_gauss",
            "fiber_coupling_doublet"]
# demos assembled entirely from anchored add() calls (no chained element) ->
# auto_tolerances stores nothing
ADD_ONLY = ["scatter_plate", "treacy_compressor"]


def _read_configs(fcstd_path):
    """(optimize_cfg, tolerance_cfg) stashed on the miewb_vars sheet."""
    out = {"optimize": None, "tolerance": None}
    xml = zipfile.ZipFile(str(fcstd_path)).read(
        "Document.xml").decode("utf8", "replace")
    for prop, key in (("miewb_optimize_config", "optimize"),
                      ("miewb_tolerance_config", "tolerance")):
        m = re.search(r'name="%s".*?<String value="([^"]*)"' % re.escape(prop),
                      xml, re.S)
        if m:
            out[key] = json.loads(html.unescape(m.group(1))).get(key)
    return out["optimize"], out["tolerance"]


def _demo_fcstds():
    return sorted(DEMOS_DIR.glob("*.FCStd"))


@pytest.mark.parametrize("fcstd", _demo_fcstds(), ids=lambda p: p.stem)
def test_demo_config_round_trips_through_panes(qtbot, fcstd):
    from mieworkbench.panes.optimize_pane import OptimizePane
    from mieworkbench.panes.tolerance_pane import TolerancePane
    opt, tol = _read_configs(fcstd)
    if opt is not None:
        pane = OptimizePane()
        qtbot.addWidget(pane)
        pane.apply_config(opt)
        back = pane.config()
        assert len(back["var"]) == len(opt["var"]), \
            "%s: optimize dropped a variable row" % fcstd.stem
        assert len(back["operand"]) == len(opt["operand"]), \
            "%s: optimize dropped an operand row" % fcstd.stem
        assert back["operand"] == opt["operand"]
    if tol is not None:
        pane = TolerancePane()
        qtbot.addWidget(pane)
        pane.apply_config(tol)
        back = pane.config()
        assert len(back["tolerance"]) == len(tol["tolerance"]), \
            "%s: tolerance dropped a row (unparseable spec)" % fcstd.stem
        assert len(back["operand"]) == len(tol["operand"])


@pytest.mark.parametrize("name", SHOWCASE)
def test_showcase_ships_optimize_config(name):
    fcstd = DEMOS_DIR / ("%s.FCStd" % name)
    if not fcstd.exists():
        pytest.skip("%s not built" % name)
    opt, _ = _read_configs(fcstd)
    assert opt and opt.get("var"), \
        "%s must ship an optimize config with variables" % name


@pytest.mark.parametrize("name", ADD_ONLY)
def test_add_only_demo_has_no_tolerance_config(name):
    fcstd = DEMOS_DIR / ("%s.FCStd" % name)
    if not fcstd.exists():
        pytest.skip("%s not built" % name)
    _, tol = _read_configs(fcstd)
    assert tol is None, \
        "%s has no chained element -> should ship no tolerance config" % name
