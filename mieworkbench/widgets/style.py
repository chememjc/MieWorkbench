"""Global widget-level stylesheet fragments.

Kept separate from mainwindow.py so any future top-level chrome (not just
MainWindow) can reuse the same rules. Applied ONCE, at the QMainWindow
level, via QWidget.setStyleSheet() -- Qt cascades widget-level stylesheets
(e.g. the stage chips' per-QLabel setStyleSheet calls) OVER a window-level
one, so this never fights more specific styling done elsewhere.
"""

from PySide6.QtGui import QPalette


def checked_toolbutton_stylesheet(palette):
    """Return a Qt stylesheet rule giving checked QToolButtons (toolbar
    toggle actions -- ray-overlay, anim-enable, face-indicators, ...) a
    visibly distinct background instead of relying solely on the
    platform style's often-subtle checked look.

    Derived from the given QPalette's Highlight color: a translucent
    tint (alpha ~60/255) as background plus a slightly stronger
    translucent border (alpha ~180/255), so it reads correctly in both
    light and dark palettes without hardcoding either.
    """
    highlight = palette.color(QPalette.ColorRole.Highlight)
    r, g, b = highlight.red(), highlight.green(), highlight.blue()
    return (
        "QToolBar QToolButton:checked {"
        " background-color: rgba(%d, %d, %d, 60);"
        " border: 1px solid rgba(%d, %d, %d, 180);"
        " border-radius: 3px;"
        "}" % (r, g, b, r, g, b)
    )
