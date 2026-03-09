"""S3 backup for TensorBoard logs via rclone.
Ref: CLAUDE.md Sec 5.2 — rsync to MinIO bucket.
"""

import subprocess
import threading
import time
from pathlib import Path


class S3BackupDaemon:
    """Background thread that periodically syncs TensorBoard logs to S3."""

    def __init__(
        self,
        local_dir: str | Path,
        remote: str = "scilib-store",
        bucket: str = "scilibmath-v2-logs",
        interval_minutes: float = 30,
    ):
        self.local_dir = Path(local_dir)
        self.remote_path = f"{remote}:{bucket}"
        self.interval = interval_minutes * 60
        self._stop = threading.Event()
        self._thread = None

    def _sync(self):
        """Run rclone sync."""
        if not self.local_dir.exists():
            return
        try:
            subprocess.run(
                ["rclone", "sync", str(self.local_dir), self.remote_path, "--quiet"],
                timeout=300,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # rclone not installed or timeout

    def _loop(self):
        while not self._stop.is_set():
            self._sync()
            self._stop.wait(self.interval)

    def start(self):
        """Start background sync daemon."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop daemon and do final sync."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._sync()  # Final sync

    def sync_now(self):
        """Force immediate sync."""
        self._sync()
