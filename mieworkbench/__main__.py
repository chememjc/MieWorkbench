import os

# The GUI must start on an unconfigured machine so the user can fix the
# paths in Settings -> Tool Paths (which edits miewb.env). CLI scripts
# still hard-fail at import; MainWindow checks common.UNCONFIGURED and
# auto-opens the Settings dialog.
os.environ.setdefault("MIEWB_ALLOW_UNCONFIGURED", "1")

from .app import main

raise SystemExit(main())
