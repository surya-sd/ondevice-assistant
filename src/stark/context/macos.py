"""
Retrieve the active application and window title via macOS Accessibility API.

Requires "Accessibility" permission in System Settings → Privacy & Security.
Degrades gracefully (returns None) when permission is denied or on non-macOS.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


class MacOSContext:
    """
    Queries the frontmost application and its focused window title.

    Example output: "VSCode — main.py"
    """

    def __init__(self) -> None:
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import AppKit  # noqa: F401
            import ApplicationServices  # noqa: F401
            return True
        except ImportError:
            log.warning("pyobjc not available — context module disabled.")
            return False

    def get(self) -> Optional[str]:
        """
        Return a short context string describing the active window, or None.

        Never raises — failures are logged at DEBUG level.
        """
        if not self._available:
            return None
        try:
            return self._query()
        except Exception as exc:
            log.debug("Context query failed: %s", exc)
            return None

    def _query(self) -> Optional[str]:
        import AppKit
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
            kAXFocusedWindowAttribute,
            kAXTitleAttribute,
        )

        # Get frontmost app
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if front_app is None:
            return None

        app_name: str = front_app.localizedName() or ""
        pid: int = front_app.processIdentifier()

        # Get focused window title via AXUIElement
        ax_app = AXUIElementCreateApplication(pid)
        err, window = AXUIElementCopyAttributeValue(ax_app, kAXFocusedWindowAttribute, None)
        if err != 0 or window is None:
            return app_name or None

        err, title = AXUIElementCopyAttributeValue(window, kAXTitleAttribute, None)
        if err != 0 or not title:
            return app_name or None

        window_title = str(title).strip()
        if window_title and window_title != app_name:
            return f"{app_name} — {window_title}"
        return app_name
