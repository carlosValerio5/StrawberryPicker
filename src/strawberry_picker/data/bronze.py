"""Bronze layer: land raw sources untouched. Contract: ``data/bronze/README.md``."""

from __future__ import annotations

import re
import shutil
import zipfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from strawberry_picker.paths import Layer, layer_dir

if TYPE_CHECKING:
    from strawberry_picker.data.sources import SourceSpec

_SOURCE_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def validate_source_name(source_name: str) -> str:
    if not _SOURCE_NAME.match(source_name):
        raise ValueError(f"source_name must be snake_case, got {source_name!r}")
    return source_name


def _pull_name(ingest_date: date, revision: int | None) -> str:
    if revision is None:
        return ingest_date.isoformat()
    if revision < 2:
        raise ValueError(
            f"revision must be >= 2 (the first pull of a day has none), got {revision}"
        )
    return f"{ingest_date.isoformat()}.{revision}"


def source_dir(source_name: str, ingest_date: date, revision: int | None = None) -> Path:
    """``data/bronze/<source>/<YYYY-MM-DD>[.<n>]/``: where one pull of a source lands.

    ``.<n>`` (n >= 2) marks a second pull on the same day, e.g. a fuller re-download.
    """
    return (
        layer_dir(Layer.BRONZE)
        / validate_source_name(source_name)
        / _pull_name(ingest_date, revision)
    )


_PULL_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\.(\d+))?$")


def all_pulls(source_name: str) -> list[Path]:
    """Every ``<date>[.<n>]`` pull of a source, oldest first."""
    root = layer_dir(Layer.BRONZE) / validate_source_name(source_name)
    pulls = []
    for path in root.glob("*") if root.is_dir() else ():
        match = _PULL_NAME.match(path.name)
        if not match:
            continue
        try:
            pulls.append((date.fromisoformat(match[1]), int(match[2] or 1), path))
        except ValueError:
            continue
    return [path for *_, path in sorted(pulls)]


def latest_pull(source_name: str) -> Path | None:
    """Newest pull by (date, revision), or ``None`` if the source was never ingested."""
    pulls = all_pulls(source_name)
    return pulls[-1] if pulls else None


def render_source_card(
    spec: SourceSpec, ingest_date: date, artifact_name: str, revision: int | None = None
) -> str:
    """Contents of the ``SOURCE.md`` every bronze pull must carry."""
    usage = "allowed" if spec.commercial_use else "**Non-commercial use only**"
    lines = [
        f"# {spec.title}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Source name | `{spec.name}` |",
        f"| URL | {spec.url} |",
        f"| DOI | {spec.doi or '—'} |",
        f"| License | {spec.license} |",
        f"| Commercial use | {usage} |",
        f"| Retrieved | {ingest_date.isoformat()}"
        + (f" (revision {revision})" if revision else "")
        + " |",
        f"| Artifact | `{artifact_name}` |",
    ]
    if spec.citation:
        lines += ["", "## Citation", "", spec.citation]
    if spec.notes:
        lines += ["", "## Notes", "", spec.notes.strip()]
    return "\n".join(lines) + "\n"


def ingest(
    spec: SourceSpec, artifact: Path, ingest_date: date, revision: int | None = None
) -> Path:
    """Land a downloaded archive or folder as a new, append-only bronze pull.

    The pull is assembled in a hidden temp folder and renamed into place, so an
    interrupted ingest never leaves a half-written pull behind.
    """
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    dest = source_dir(spec.name, ingest_date, revision)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists — bronze is append-only")
    partial = dest.with_name(f".{dest.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    if artifact.is_dir():
        shutil.copytree(artifact, partial)
    elif zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            archive.extractall(partial)  # strips absolute paths and ".." components
    else:
        partial.mkdir(parents=True)
        shutil.unpack_archive(artifact, partial, filter="data")  # tar: refuse unsafe members
    card = render_source_card(spec, ingest_date, artifact.name, revision)
    (partial / "SOURCE.md").write_text(card)
    partial.rename(dest)
    return dest
