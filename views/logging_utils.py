import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional


class FastViewLogger:
    """Utility logger that only emits output when fast processing is enabled."""

    def __init__(self, st_module, view_name: str) -> None:
        self.st = st_module
        self.view_name = view_name
        self.enabled = False
        try:
            self.enabled = bool(st_module.session_state.get("use_fast_processing", False))
        except Exception:
            self.enabled = False

    def log(self, message: str, **extra: Any) -> None:
        if not self.enabled:
            return
        timestamp = time.strftime("%H:%M:%S")
        details = " ".join(f"{key}={self._format(value)}" for key, value in extra.items())
        print(f"[FAST][{self.view_name}] {timestamp} {message}" + (f" | {details}" if details else ""))

    @contextmanager
    def time_block(self, label: str, **meta: Any):
        start = time.perf_counter()
        self.log(f"⏳ {label} started", **meta)
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.log(f"✅ {label} finished", duration=f"{duration:.3f}s", **meta)

    def log_dataframe(
        self,
        label: str,
        df,
        *,
        include_dtypes: bool = False,
        top_columns: int = 8,
    ) -> None:
        if not self.enabled or df is None:
            return
        try:
            rows = len(df)
            cols = len(getattr(df, "columns", []))
        except Exception:
            rows = getattr(df, "shape", (0,))[0]
            cols = getattr(df, "shape", (0, 0))[1]
        info: Dict[str, Any] = {"rows": rows, "cols": cols}
        if cols:
            columns = list(df.columns)[:top_columns]
            info["columns"] = columns
        if include_dtypes:
            try:
                dtype_items = list(df.dtypes.astype(str).items())[:top_columns]
                info["dtypes"] = dtype_items
            except Exception:
                pass
        mem_bytes: Optional[int] = None
        try:
            mem_bytes = int(df.memory_usage(deep=True).sum())
            info["memory_bytes"] = mem_bytes
        except Exception:
            pass
        self.log(f"{label}", **info)

    def log_total(self, label: str, start_time: float, **meta: Any) -> None:
        if not self.enabled:
            return
        duration = time.perf_counter() - start_time
        self.log(label, duration=f"{duration:.3f}s", **meta)

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (list, tuple)):
            preview = ",".join(str(v) for v in value[:5])
            if len(value) > 5:
                preview += "…"
            return f"[{preview}]"
        if isinstance(value, dict):
            items = list(value.items())[:5]
            preview = ",".join(f"{k}:{v}" for k, v in items)
            if len(value) > 5:
                preview += "…"
            return f"{{{preview}}}"
        return str(value)
