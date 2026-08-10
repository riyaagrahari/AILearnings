from pathlib import Path

class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def event(self, message, passed=True):
        line = f"[{'PASS' if passed else 'FAIL'}] {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
