from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


class FileOps:

    def __init__(self, base_dir: str = ".") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, path: str, content: str) -> bool:
        try:
            full = self.base_dir / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False

    def read(self, path: str) -> Optional[str]:
        try:
            return (self.base_dir / path).read_text(encoding="utf-8")
        except Exception:
            return None

    def write_json(self, path: str, data: Any) -> bool:
        return self.write(path, json.dumps(data, indent=2, default=str))

    def read_json(self, path: str) -> Optional[Any]:
        content = self.read(path)
        if content:
            try:
                return json.loads(content)
            except Exception:
                return None
        return None

    def exists(self, path: str) -> bool:
        return (self.base_dir / path).exists()

    def list_files(self, subdir: str = "", ext: str = "") -> list:
        target = self.base_dir / subdir if subdir else self.base_dir
        if not target.exists():
            return []
        files = list(target.rglob(f"*{ext}") if ext else target.rglob("*"))
        return [str(f.relative_to(self.base_dir)) for f in files if f.is_file()]

    def delete(self, path: str) -> bool:
        try:
            (self.base_dir / path).unlink()
            return True
        except Exception:
            return False