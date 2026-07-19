"""PreviewConfigWidget - per-document ray-preview pattern editor (WP2).

A small kind combo (Fan / Rings) driving a QStackedWidget of param
editors, backed by scripts/common.parse_viz_pattern_spec so the widget
can never produce (or accept) a spec the pipeline/preview chain would
reject. Two pure module-level helpers do the actual spec<->fields
translation so they can be unit-tested without a QApplication:

    spec_from_fields(kind, **kw) -> "fan:n=5" | "rings:dr=1:nper=8[:nrings=K]"
    fields_from_spec(spec)       -> {"kind": "fan", "n": 5} | {...}

The widget itself is dumb state (spec() / set_spec()) -- callers
(MainWindow's Simulation Settings dialog) own persistence.
"""

import os
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
import common  # noqa: E402  (stdlib-only shared contract hub)

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

_KINDS = ("fan", "rings")


def spec_from_fields(kind, **kw):
    """Build a --viz-pattern spec string from widget-shaped fields.

    fan:   n=<int>
    rings: dr_mm=<float>, nper=<int>, nrings=<int, 0 means "omit -> auto">
    """
    kind = str(kind).strip().lower()
    if kind == "fan":
        return "fan:n=%d" % int(kw.get("n", 5))
    if kind == "rings":
        spec = "rings:dr=%g:nper=%d" % (
            float(kw["dr_mm"]), int(kw["nper"]))
        nrings = int(kw.get("nrings", 0) or 0)
        if nrings > 0:
            spec += ":nrings=%d" % nrings
        return spec
    raise ValueError("unknown viz-pattern kind %r" % kind)


def fields_from_spec(spec):
    """spec -> widget-shaped fields dict (raises ValueError via
    common.parse_viz_pattern_spec on a malformed spec)."""
    parsed = common.parse_viz_pattern_spec(spec)
    if parsed["kind"] == "fan":
        return {"kind": "fan", "n": parsed["n"]}
    return {"kind": "rings", "dr_mm": parsed["dr_mm"],
            "nper": parsed["nper"], "nrings": parsed["nrings"] or 0}


class PreviewConfigWidget(QWidget):
    """Kind combo + stacked param editors for a single --viz-pattern
    spec. spec()/set_spec() are the whole API; the widget holds no
    project/document reference of its own. specChanged fires with the
    freshly composed spec on every user edit (and, incidentally, during
    set_spec -- listeners that echo the spec elsewhere guard reentrancy
    themselves)."""

    specChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Fan", "fan")
        self.kind_combo.addItem("Rings", "rings")
        outer.addWidget(self.kind_combo)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        fan_page = QWidget()
        fan_form = QFormLayout(fan_page)
        self.fan_n_spin = QSpinBox()
        self.fan_n_spin.setRange(1, 999)
        self.fan_n_spin.setValue(5)
        self.fan_n_spin.setToolTip(
            "Rays per source: center + edge midpoints, then rim fill")
        fan_form.addRow("Rays per source", self.fan_n_spin)
        self.stack.addWidget(fan_page)

        rings_page = QWidget()
        rings_form = QFormLayout(rings_page)
        self.rings_dr_spin = QDoubleSpinBox()
        self.rings_dr_spin.setRange(0.001, 1000.0)
        self.rings_dr_spin.setDecimals(3)
        self.rings_dr_spin.setSuffix(" mm")
        self.rings_dr_spin.setValue(1.0)
        rings_form.addRow("Ring spacing", self.rings_dr_spin)
        self.rings_nper_spin = QSpinBox()
        self.rings_nper_spin.setRange(1, 999)
        self.rings_nper_spin.setValue(12)
        rings_form.addRow("Rays per ring", self.rings_nper_spin)
        self.rings_nrings_spin = QSpinBox()
        self.rings_nrings_spin.setRange(0, 999)
        self.rings_nrings_spin.setValue(0)
        self.rings_nrings_spin.setSpecialValueText("Auto")
        self.rings_nrings_spin.setToolTip(
            "Number of rings to fill the emit face; 0 = auto")
        rings_form.addRow("Rings", self.rings_nrings_spin)
        self.stack.addWidget(rings_page)

        self.kind_combo.currentIndexChanged.connect(
            self.stack.setCurrentIndex)

        self.kind_combo.currentIndexChanged.connect(self._emit_spec)
        for spin in (self.fan_n_spin, self.rings_dr_spin,
                     self.rings_nper_spin, self.rings_nrings_spin):
            spin.valueChanged.connect(self._emit_spec)

    def _emit_spec(self, *_args):
        self.specChanged.emit(self.spec())

    # -- API --------------------------------------------------------------------
    def spec(self):
        """The current widget state as a --viz-pattern spec string."""
        kind = self.kind_combo.currentData()
        if kind == "fan":
            return spec_from_fields("fan", n=self.fan_n_spin.value())
        return spec_from_fields(
            "rings", dr_mm=self.rings_dr_spin.value(),
            nper=self.rings_nper_spin.value(),
            nrings=self.rings_nrings_spin.value())

    def set_spec(self, spec):
        """Populate the widget from a spec string. Raises ValueError on
        a malformed spec (via fields_from_spec) -- the widget state is
        left unchanged in that case."""
        fields = fields_from_spec(spec)
        idx = _KINDS.index(fields["kind"])
        self.kind_combo.setCurrentIndex(idx)
        self.stack.setCurrentIndex(idx)
        if fields["kind"] == "fan":
            self.fan_n_spin.setValue(fields["n"])
        else:
            self.rings_dr_spin.setValue(fields["dr_mm"])
            self.rings_nper_spin.setValue(fields["nper"])
            self.rings_nrings_spin.setValue(fields["nrings"])
