"""Plot data inspection (WP3): pure, reusable helpers shared by the
Optimize and Tolerance panes, plus a non-modal data-table dialog and a
dialog-free context-menu builder.

Everything above the Qt classes is Qt-free and directly unit-testable:

  * dense_ranks()          - dense (1,2,2,3) ranking of merits, ascending
  * nearest_point_index()  - pure pixel hit test for hover
  * format_point_tooltip() - the multi-line tooltip string
  * history_table()        - (headers, rows) for the three data-table kinds

The two Qt classes are the shell over those functions:

  * DataTableDialog        - a NON-modal QTableWidget populated from
                             history_table with an "Export CSV…" button
                             (reuses widgets/table_export.export_table_csv)
  * build_plot_menu()      - build (don't show) a right-click QMenu

The lower two mirror panes/results.py: the ONE save QFileDialog lives
behind a monkeypatchable module seam (_choose_csv_path), and the dialog
is non-modal + WA_DeleteOnClose False so an offscreen test never blocks
and the owner keeps a live reference (CLAUDE.md: never an unguarded modal
in a pane code path).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMenu, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from ..widgets.table_export import export_table_csv

# merit sentinel: optimize.PENALTY — such an eval/draw has no real merit
PENALTY_FLOOR = 1e8


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def dense_ranks(merits, penalized_floor=PENALTY_FLOOR):
    """DENSE ranking (1, 2, 2, 3) of `merits` by ascending value (smaller
    merit ranks better = 1). Ties share a rank; the next distinct value
    takes the next integer. A merit that is None or >= penalized_floor is
    unranked (None). Returns a list aligned to the input order."""
    distinct = sorted({m for m in merits
                       if m is not None and m < penalized_floor})
    rank_of = {v: i + 1 for i, v in enumerate(distinct)}
    return [None if (m is None or m >= penalized_floor) else rank_of[m]
            for m in merits]


def nearest_point_index(points_px, pos_px, max_dist_px=12):
    """Index of the point in `points_px` (an iterable of (x, y) pixels)
    nearest to `pos_px`, or None when the nearest is farther than
    max_dist_px. Pure Euclidean hit test — no Qt."""
    px, py = pos_px
    lim2 = float(max_dist_px) * float(max_dist_px)
    best_i, best_d2 = None, None
    for i, (x, y) in enumerate(points_px):
        d2 = (x - px) ** 2 + (y - py) ** 2
        if d2 <= lim2 and (best_d2 is None or d2 < best_d2):
            best_i, best_d2 = i, d2
    return best_i


def _fmt(value):
    """Compact cell/tooltip formatting: '' for None, %.6g for floats,
    str() otherwise."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def format_point_tooltip(label, params, merit, rank):
    """A hover tooltip like::

        eval 12 — merit 3.4e-3 (rank 2)
        z = 94.5
        efl = 200

    `label` is the head (e.g. "eval 12"); merit/rank are appended when
    not None; each params item becomes a "k = v" line (insertion order
    preserved)."""
    head = str(label)
    if merit is not None:
        head += " — merit %.4g" % merit
    if rank is not None:
        head += " (rank %d)" % rank
    lines = [head]
    for key, val in (params or {}).items():
        lines.append("%s = %s" % (key, _fmt(val)))
    return "\n".join(lines)


def _var_union(entries):
    """Insertion-ordered union of every entry's params keys."""
    seen = []
    for entry in entries:
        for key in (entry.get("params") or {}):
            if key not in seen:
                seen.append(key)
    return seen


def history_table(entries, kind):
    """(headers, rows) for a DataTableDialog. `entries` is the pane's
    stored history; `kind` selects the layout:

      * "optimize"    -> eval#  | <var…> | merit | rank
      * "mc"          -> draw#  | <var…> | merit | pass/fail | rank
      * "sensitivity" -> name | impact | derivative | rank

    Var columns are the insertion-ordered union of every entry's param
    keys. `rows` is a list of lists of already-formatted strings."""
    entries = list(entries or [])
    if kind == "sensitivity":
        headers = ["name", "impact", "derivative", "rank"]
        rows = [[_fmt(e.get("name")), _fmt(e.get("impact")),
                 _fmt(e.get("derivative")), _fmt(e.get("rank"))]
                for e in entries]
        return headers, rows

    varnames = _var_union(entries)
    ranks = dense_ranks([e.get("merit") for e in entries])
    if kind == "optimize":
        headers = ["eval#"] + varnames + ["merit", "rank"]
        idkey = "eval"
    elif kind == "mc":
        headers = ["draw#"] + varnames + ["merit", "pass/fail", "rank"]
        idkey = "draw"
    else:
        raise ValueError("unknown history_table kind %r" % (kind,))

    rows = []
    for entry, rank in zip(entries, ranks):
        params = entry.get("params") or {}
        row = [_fmt(entry.get(idkey))]
        row += [_fmt(params.get(name)) for name in varnames]
        row.append(_fmt(entry.get("merit")))
        if kind == "mc":
            passed = entry.get("passed")
            row.append("" if passed is None
                       else ("pass" if passed else "fail"))
        row.append("" if rank is None else str(rank))
        rows.append(row)
    return headers, rows


# ---------------------------------------------------------------------------
# Qt shell
# ---------------------------------------------------------------------------
def _choose_csv_path(parent, default_name):
    """Dialog seam: the ONE place a save QFileDialog is shown for CSV
    export. Tests monkeypatch this module function so offscreen runs
    never open a real modal (mirrors results._choose_save_path)."""
    path, _filt = QFileDialog.getSaveFileName(
        parent, "Export data as CSV…", default_name, "CSV (*.csv)")
    return path or None


def build_plot_menu(callbacks, parent=None):
    """Build (don't show) a right-click QMenu from `callbacks`, an
    iterable of (label, callable). Dialog-free and testable — the caller
    exec()s it. Each action's triggered signal calls the callable."""
    menu = QMenu(parent)
    for label, callback in callbacks:
        action = menu.addAction(str(label))
        action.triggered.connect(callback)
    return menu


class DataTableDialog(QDialog):
    """Non-modal table view of a plot's underlying data. Populated from
    history_table(entries, kind); an "Export CSV…" button writes the
    table through widgets/table_export.export_table_csv."""

    def __init__(self, entries, kind, title="Data", parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setWindowTitle(title)

        headers, rows = history_table(entries, kind)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(rows), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setToolTip("Write this table to a CSV file")
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

    def export_to(self, path):
        """Write the table to `path` (the test seam — no dialog)."""
        export_table_csv(self.table, path)
        return path

    def _on_export(self):
        path = _choose_csv_path(self, "plot_data.csv")
        if path:
            self.export_to(path)
