from pathlib import Path

def load_team(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Team file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Team file is empty: {path}")
    return text
