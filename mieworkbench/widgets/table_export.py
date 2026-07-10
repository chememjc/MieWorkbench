"""CSV export for QTableWidget-based displays (Results/Compare panes).

One function, used by both results.py's summary/power tables and
compare_pane.py's metrics_table. Atomic write (tmp file + os.replace),
matching the house pattern in panes/prop_editor.py's
_atomic_write_registry -- a crash/kill mid-write must never leave a
half-written CSV at the destination path.
"""

import csv
import os


def export_table_csv(table, path):
    """Write a QTableWidget's header + all cell text to CSV at path.

    Header comes from horizontalHeaderItem(c).text() (empty string if a
    column has no header item); cells come from item(r, c).text() (empty
    string for an empty/absent item -- QTableWidgetItem may be None for
    a cell nothing ever set)."""
    ncols = table.columnCount()
    header = [
        (table.horizontalHeaderItem(c).text()
         if table.horizontalHeaderItem(c) is not None else "")
        for c in range(ncols)
    ]
    rows = []
    for r in range(table.rowCount()):
        row = []
        for c in range(ncols):
            item = table.item(r, c)
            row.append(item.text() if item is not None else "")
        rows.append(row)

    path = str(path)
    dest_dir = os.path.dirname(path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)
