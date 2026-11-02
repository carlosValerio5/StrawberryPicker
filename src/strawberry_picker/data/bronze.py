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


def source_dir(source_name: str, ingest_date: date) -> Path:
    """``data/bronze/<source_name>/<YYYY-MM-DD>/`` — where one pull of a source lands."""
    return layer_dir(Layer.BRONZE) / validate_source_name(source_name) / ingest_date.isoformat()


def latest_pull(source_name: str) -> Path | None:
    """Most recent ``<YYYY-MM-DD>`` pull of a source, or ``None`` if it was never ingested."""
    root = layer_dir(Layer.BRONZE) / validate_source_name(source_name)
    pulls = []
    for path in root.glob("*") if root.is_dir() else ():
        try:
            pulls.append((date.fromisoformat(path.name), path))
        except ValueError:
            continue
    return max(pulls)[1] if pulls else None


def render_source_card(spec: SourceSpec, ingest_date: date, artifact_name: str) -> str:
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
        f"| Retrieved | {ingest_date.isoformat()} |",
        f"| Artifact | `{artifact_name}` |",
    ]
    if spec.citation:
        lines += ["", "## Citation", "", spec.citation]
    if spec.notes:
        lines += ["", "## Notes", "", spec.notes.strip()]
    return "\n".join(lines) + "\n"


def ingest(spec: SourceSpec, artifact: Path, ingest_date: date) -> Path:
    """Land a downloaded archive or folder as a new, append-only bronze pull.

    The pull is assembled in a hidden temp folder and renamed into place, so an
    interrupted ingest never leaves a half-written pull behind.
    """
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    dest = source_dir(spec.name, ingest_date)
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
    (partial / "SOURCE.md").write_text(render_source_card(spec, ingest_date, artifact.name))
    partial.rename(dest)
    return dest
