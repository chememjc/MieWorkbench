"""LibraryPane tests (offscreen). Reads the real primitives/ tree
read-only for listing; a user-dropped-file scenario uses a tmp copy so the
real tree is never written to. The "Add to scene" default-label / label
dialog logic is driven as plain functions/methods, never via exec()."""
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from mieworkbench.core.librarymgr import LibraryManager    # noqa: E402
from mieworkbench.core.proplib import CATEGORIES           # noqa: E402
from mieworkbench.panes.library import LibraryPane, default_label  # noqa: E402

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "opticalproperties"))
PRIMITIVES_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "primitives"))


def test_default_label_running_number():
    assert default_label("lens_pcx", set()) == "lens_pcx"
    assert default_label("lens_pcx", {"lens_pcx"}) == "lens_pcx_2"
    assert default_label("lens_pcx", {"lens_pcx", "lens_pcx_2"}) \
        == "lens_pcx_3"
    assert default_label("lens_pcx", {"lens_pcx", "lens_pcx_3"}) \
        == "lens_pcx_2"


def test_elements_tab_lists_at_least_20_primitives_grouped(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)

    assert pane.primitive_count() >= 20
    assert pane.tree.topLevelItemCount() >= 1

    total_children = sum(
        pane.tree.topLevelItem(i).childCount()
        for i in range(pane.tree.topLevelItemCount()))
    assert total_children == pane.primitive_count()

    categories = {pane.tree.topLevelItem(i).text(0)
                 for i in range(pane.tree.topLevelItemCount())}
    assert "Lenses" in categories


def test_bladed_iris_primitive_listed_under_apertures(qtbot):
    # engine3 Sec 11 / P8: the N-blade iris ships as a catalog primitive
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    item = next((i for i in pane._primitives if i["kind"] == "iris_bladed"),
                None)
    assert item is not None, "iris_bladed missing from the element catalog"
    assert item["category"] == "Apertures"
    assert "n_blades" in item["params"]
    categories = {pane.tree.topLevelItem(i).text(0)
                  for i in range(pane.tree.topLevelItemCount())}
    assert "Apertures" in categories


def test_tabs_present(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    labels = [pane.tabs.tabText(i) for i in range(pane.tabs.count())]
    assert labels == ["Elements", "Project library", "System library"]


def test_add_element_requested_payload(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)

    lens_item = next(i for i in pane._primitives if i["kind"] == "lens_pcx")

    with qtbot.waitSignal(pane.addElementRequested, timeout=1000) as blocker:
        pane.request_add_element(lens_item, "lens_pcx")
    info, label = blocker.args
    assert info["kind"] == "lens_pcx"
    assert label == "lens_pcx"
    assert info["params"]["R_front"]["default"] == 25.0


def test_add_element_running_label_across_calls(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    lens_item = next(i for i in pane._primitives if i["kind"] == "lens_pcx")

    label1 = default_label(lens_item["kind"], pane._used_labels)
    pane.request_add_element(lens_item, label1)
    label2 = default_label(lens_item["kind"], pane._used_labels)

    assert label1 == "lens_pcx"
    assert label2 == "lens_pcx_2"


def test_selection_updates_details_box(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)

    group = pane.tree.topLevelItem(0)
    child = group.child(0)
    pane.tree.setCurrentItem(child)
    assert pane.details.text() != ""


def test_project_summary_disabled_without_project_root(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    assert not pane.project_summary.open_button.isEnabled()


def test_system_summary_counts(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    texts = [pane.system_summary.list_widget.item(i).text()
            for i in range(pane.system_summary.list_widget.count())]
    # rows are just "<category> (<count>)" -- no entry names (too noisy).
    # Compare against the real registry row counts (not hard-coded magic
    # numbers) so this stays correct as rows are added to the library.
    for category in CATEGORIES:
        expected = len(mgr.system_lib.registry_rows(category))
        assert "%s (%d)" % (category, expected) in texts


def test_summary_rows_have_no_entry_names(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    for i in range(pane.system_summary.list_widget.count()):
        text = pane.system_summary.list_widget.item(i).text()
        assert ":" not in text
        assert text.split(" (")[0] in CATEGORIES


def test_open_editor_requested_from_summary(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    pane.system_summary.list_widget.setCurrentRow(0)
    with qtbot.waitSignal(pane.openEditorRequested, timeout=1000) as blocker:
        pane.system_summary._on_open_clicked()
    assert blocker.args == ["materials", "system"]


def test_double_click_summary_row_emits_open_editor_requested(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    item = pane.system_summary.list_widget.item(1)   # "coatings (10)"
    assert item.text().startswith("coatings")
    with qtbot.waitSignal(pane.openEditorRequested, timeout=1000) as blocker:
        pane.system_summary._on_item_double_clicked(item)
    assert blocker.args == ["coatings", "system"]


def test_double_click_project_summary_row_emits_via_library_pane(qtbot):
    mgr = LibraryManager(REPO_ROOT, PRIMITIVES_ROOT)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    # project library disabled without a project root: emit still routes
    # through the same LibraryPane-level signal, using CATEGORIES[0]
    item = pane.project_summary.list_widget.item(0)
    with qtbot.waitSignal(pane.openEditorRequested, timeout=1000) as blocker:
        pane.project_summary._on_item_double_clicked(item)
    assert blocker.args == [CATEGORIES[0], "project"]


def test_refresh_picks_up_user_dropped_fcstd(qtbot, tmp_path):
    prims_dir = tmp_path / "primitives"
    shutil.copytree(PRIMITIVES_ROOT, prims_dir)
    mgr = LibraryManager(REPO_ROOT, prims_dir)
    pane = LibraryPane(mgr)
    qtbot.addWidget(pane)
    before = pane.primitive_count()

    (prims_dir / "user_widget.FCStd").write_bytes(b"not real")
    pane.refresh()

    assert pane.primitive_count() == before + 1
    user_item = next(i for i in pane._primitives
                     if i["kind"] == "user_widget")
    assert user_item["category"] == "User"
