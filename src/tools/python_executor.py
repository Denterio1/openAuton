from __future__ import annotations
import subprocess, sys, tempfile, os, time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    code_ran: str

    def short(self) -> str:
        status = "✅" if self.success else "❌"
        lines = self.stdout.strip().split("\n")
        preview = lines[-1][:100] if lines else ""
        return f"{status} [{self.duration_seconds:.2f}s] {preview}"


class PythonExecutor:

    def __init__(self, timeout: int = 30, max_output: int = 5000) -> None:
        self.timeout    = timeout
        self.max_output = max_output

    def run(self, code: str) -> ExecutionResult:
        start = time.time()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:self.max_output],
                stderr=result.stderr[:self.max_output],
                return_code=result.returncode,
                duration_seconds=time.time() - start,
                code_ran=code[:500],
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, "", f"TimeoutError: exceeded {self.timeout}s", -1, time.time()-start, code[:500])
        except Exception as e:
            return ExecutionResult(False, "", str(e), -1, time.time()-start, code[:500])
        finally:
            os.unlink(tmp_path)

    def run_and_capture(self, code: str) -> str:
        result = self.run(code)
        return result.stdout if result.success else f"ERROR: {result.stderr}"

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)