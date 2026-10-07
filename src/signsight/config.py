"""Settings from environment variables. signsight needs no credentials."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ("hog", "cnn")


class ConfigError(ValueError):
    """An environment variable has a value that the code cannot use."""


def load_dotenv(path: str | Path = ".env") -> None:
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    seed: int = 42
    image_size: int = 32
    backend: str = "hog"
    epochs: int = 15
    seeds: int = 3

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ConfigError(f"SIGNSIGHT_BACKEND must be one of {BACKENDS}, got {self.backend!r}")
        if self.image_size < 16 or self.image_size % 8:
            raise ConfigError("SIGNSIGHT_IMAGE_SIZE must be a multiple of 8 and at least 16")
        if self.epochs < 1 or self.seeds < 1:
            raise ConfigError("SIGNSIGHT_EPOCHS and SIGNSIGHT_SEEDS must be at least 1")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(_get("SIGNSIGHT_DATA_DIR", "data")),
            out_dir=Path(_get("SIGNSIGHT_OUT", "runs")),
            seed=_int("SIGNSIGHT_SEED", 42),
            image_size=_int("SIGNSIGHT_IMAGE_SIZE", 32),
            backend=_get("SIGNSIGHT_BACKEND", "hog"),
            epochs=_int("SIGNSIGHT_EPOCHS", 15),
            seeds=_int("SIGNSIGHT_SEEDS", 3),
        )
