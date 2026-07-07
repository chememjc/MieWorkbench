"""Problems pane: pre-run validation findings, click-to-locate.

Runs core.validation.Validator against the live Project + property
library + the config matrix's current values; errors gate the Run button
(the main window listens to validationChanged)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from ..core import validation

_ICON = {validation.ERROR: "⛔", validation.WARNING: "⚠",
         validation.INFO: "ℹ"}


class ProblemsPane(QWidget):
    selectBodyRequested = Signal(str)      # body name to highlight
    validationChanged = Signal(bool)       # True = has blocking errors

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.optprops_loader = None        # callable -> OpticalProperties
        self.config_provider = None        # callable -> dict

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.btn = QPushButton("Validate scene")
        self.btn.setToolTip("Check the scene for missing information and "
                            "likely errors before running")
        self.btn.clicked.connect(self.run_checks)
        row.addWidget(self.btn)
        self.deep = QPushButton("Deep check")
        self.deep.setToolTip("Also run FreeCAD-side geometry checks "
                             "(recompute errors, open solids, overlaps)")
        self.deep.clicked.connect(lambda: self.run_checks(deep=True))
        row.addWidget(self.deep)
        self.summary = QLabel("")
        row.addWidget(self.summary, 1)
        lay.addLayout(row)

        self.listw = QListWidget()
        self.listw.setToolTip("Double-click a finding to select the "
                              "affected element")
        self.listw.itemDoubleClicked.connect(self._activate)
        lay.addWidget(self.listw)

    def set_project(self, project):
        self.project = project

    def set_sources(self, optprops_loader, config_provider):
        self.optprops_loader = optprops_loader
        self.config_provider = config_provider

    def run_checks(self, deep=False):
        self.listw.clear()
        if self.project is None or not self.project.is_open():
            self.summary.setText("no model open")
            self.validationChanged.emit(False)
            return []
        optprops = None
        if self.optprops_loader is not None:
            try:
                optprops = self.optprops_loader()
            except Exception:
                optprops = None
        config = {}
        if self.config_provider is not None:
            try:
                config = self.config_provider() or {}
            except Exception:
                config = {}
        v = validation.Validator(self.project.structure, optprops, config)
        findings = v.validate()
        if deep:
            try:
                findings += validation.run_deep_checks(self.project)
            except Exception as exc:
                findings.append(validation.Finding(
                    validation.WARNING, "deep check failed: %s" % exc,
                    check="deep-geometry"))
        for f in findings:
            item = QListWidgetItem("%s  %s" % (_ICON.get(f.severity, "•"),
                                               f.message))
            tip = f.fix_hint or ""
            if tip:
                item.setToolTip("Fix: " + tip)
            item.setData(Qt.UserRole, f.body)
            self.listw.addItem(item)
        n_err = sum(1 for f in findings
                    if f.severity == validation.ERROR)
        n_warn = sum(1 for f in findings
                     if f.severity == validation.WARNING)
        self.summary.setText("%d error(s), %d warning(s)" % (n_err, n_warn))
        self.validationChanged.emit(n_err > 0)
        return findings

    def _activate(self, item):
        body = item.data(Qt.UserRole)
        if body:
            self.selectBodyRequested.emit(body)
