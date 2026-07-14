import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    version: str
    content_hash: str
    content: str


def load_system_prompt() -> RenderedPrompt:
    root = Path(__file__).parent
    manifest = tomllib.loads((root / "manifest.toml").read_text(encoding="utf-8"))
    parts = [(root / str(manifest["system"])).read_text(encoding="utf-8")]
    parts.extend(
        (root / str(fragment)).read_text(encoding="utf-8")
        for fragment in manifest.get("fragments", [])
    )
    content = "\n\n".join(part.strip() for part in parts if part.strip())
    return RenderedPrompt(
        version=str(manifest["version"]),
        content_hash=sha256(content.encode("utf-8")).hexdigest()[:16],
        content=content,
    )
