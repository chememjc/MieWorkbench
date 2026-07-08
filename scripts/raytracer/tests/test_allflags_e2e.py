# =============================================================================
# test_allflags_e2e.py — EVERYTHING ON AT ONCE.
#
# One composite scene exercising every new physics module simultaneously,
# traced through the real run_trace.main() with all fidelity flags enabled:
#   unpolarized broadband COHERENT source (2 pol strata x 3 lambda strata)
#   -> calcite slab (birefringent o/e split, frosted side face roughness)
#   -> film polarizer (vendor table diattenuator)
#   -> bulk bandpass filter slab with a 50:50 tabulated coating on its exit
#      face and a lamellar grating on one side face
#   -> detector
# with --ray-differentials --gather-occlusion --save-fields, micro
# roughness Fresnel and polarized Mie azimuth defaults.
#
# Asserts: energy closure < 1e-3, every (source, lam, pol) gather key
# renders, no NaNs in the stored cube, fields/ groups written, occlusion +
# differential diagnostics present. A second test drives the FreeCAD-
# authored mesh_freeform scene (BVH mesh tracing) through the same flags.
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

import common                                             # noqa: E402
from raytracer.tests import scenehelpers as sh            # noqa: E402


def _composite_model():
    src = sh.source_body(power_mW=5.0, coherent=True, half=5e-4,
                         lambdac_nm=550.0, lambdamin_nm=500.0,
                         lambdamax_nm=600.0, polarization=None, x=-0.02)
    calcite = sh.slab_body(
        "Cal", "calcite", 0.000, 0.006, half=0.008,
        crystal_axis=[np.sqrt(0.5), 0.0, np.sqrt(0.5)],
        roughness_faces={"Cal.Pad.Face3": "150:lcorr=8"})
    pol = sh.slab_body(
        "Pol", "pmma", 0.008, 0.010, half=0.008,
        polarizer="thorlabs_lpvise100a", polarizer_axis=[0.0, 0.0, 1.0])
    filt = sh.slab_body(
        "Filt", "bk7", 0.012, 0.0155, half=0.008,
        filter="bp_550_40",
        coating={"Filt.Pad.Face2": "bs_5050_vis_45"},
        grating={"Filt.Pad.Face4": "600:v:orders=-1..1"})
    det = sh.detector_body(x=0.030, half=0.012)
    return sh.make_model([src, calcite, pol, filt, det])


@pytest.mark.slow
def test_allflags_composite(tmp_path):
    model = _composite_model()
    common.validate_model(model)
    mj = tmp_path / "model.json"
    mj.write_text(json.dumps(model))
    case = tmp_path / "case"

    import run_trace
    rc = run_trace.main([
        "--model-json", str(mj), "--case-dir", str(case),
        "--rays", "40000", "--nlambda", "3", "--resolution", "256",
        "--power-floor", "1e-8",
        "--ray-differentials", "--gather-occlusion", "--save-fields",
        "--rough-fresnel", "micro", "--no-gather-gate",
        "--viz-density", "2.0",
    ])
    assert rc == 0, "energy closure gate failed (exit 3)"

    audit = json.loads((case / "audit.json").read_text())
    for name, s in audit["per_seed"][0]["sources"].items():
        assert s["closure_error"] < 1e-3, (name, s["closure_error"])
        # the crossed part of the unpolarized light must land in the
        # polarizer bucket, birefringent/coating losses in the surface one
        assert s["polarizer_absorbed"] > 0.0
        assert s["absorbed_surface"] > 0.0
        assert s["absorbed_bulk"] > 0.0          # filter + media

    cj = json.loads((case / "case.json").read_text())
    gd = cj["gather"]["seed42"]
    assert gd, "no coherent gather ran"
    (label, keys), = gd.items()
    # 3 lambda strata x 2 pol strata must all have rendered
    assert len(keys) == 6, sorted(keys)
    for k, v in keys.items():
        assert v["n_samples"] > 0
        assert "occlusion" in v
        assert v["occlusion"]["n_faces_tested"] > 0
        assert v["n_differential_dA"] >= 0       # present (may be 0 after
        #                                          the birefringent split)

    import h5py
    (h5file,) = list((case / "detectors").glob("*.h5"))
    with h5py.File(h5file) as h:
        cube = h["spectral_cube_mean"][...]
        assert np.all(np.isfinite(cube)), "NaN/inf in the detector cube"
        assert cube.sum() > 0.0
        assert "fields" in h, "--save-fields wrote no fields/ group"
        fkeys = list(h["fields"].keys())
        assert len(fkeys) == 6, fkeys
        for fk in fkeys:
            Ex = h["fields"][fk]["Ex"][...]
            assert Ex.dtype == np.complex128
            assert np.all(np.isfinite(Ex.real))

    rays = np.load(case / "rays.npy")
    assert rays.shape[1] == 11
    assert np.any(rays[:, 9] == 1.0), "no e-rays recorded in viz output"
    assert np.all((rays[:, 10] >= 0.0) & (rays[:, 10] <= 1.0)), \
        "rel_power outside [0,1]"


MESH_MODEL = SCRIPTS.parent / "geometry" / "mesh_freeform" / "model.json"


@pytest.mark.slow
@pytest.mark.skipif(not MESH_MODEL.exists(),
                    reason="geometry/mesh_freeform not extracted")
def test_allflags_mesh_scene(tmp_path):
    """BVH mesh tracing under the full flag set (the composite scene above
    is analytic-only; this covers the mesh path)."""
    import run_trace
    case = tmp_path / "case"
    rc = run_trace.main([
        "--model-json", str(MESH_MODEL), "--case-dir", str(case),
        "--rays", "20000", "--nlambda", "2", "--resolution", "128",
        "--ray-differentials", "--gather-occlusion", "--save-fields",
        "--no-gather-gate", "--power-floor", "1e-8",
    ])
    assert rc == 0
    audit = json.loads((case / "audit.json").read_text())
    for name, s in audit["per_seed"][0]["sources"].items():
        assert s["closure_error"] < 1e-3, (name, s["closure_error"])
    import h5py
    (h5file,) = list((case / "detectors").glob("*.h5"))
    with h5py.File(h5file) as h:
        cube = h["spectral_cube_mean"][...]
        assert np.all(np.isfinite(cube))
        assert cube.sum() > 0.0
