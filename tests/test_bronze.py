from __future__ import annotations

import zipfile
from datetime import date

import pytest

from strawberry_picker.data.bronze import ingest, latest_pull, render_source_card


def test_latest_pull_picks_newest_date(tmp_data_dir) -> None:
    source = tmp_data_dir / "bronze" / "toy_source"
    for name in ("2026-10-01", "2026-11-02", "notes", "2026-13-40"):
        (source / name).mkdir(parents=True)
    assert latest_pull("toy_source") == source / "2026-11-02"


def test_latest_pull_none_when_missing(tmp_data_dir) -> None:
    assert latest_pull("toy_source") is None
    (tmp_data_dir / "bronze" / "toy_source").mkdir(parents=True)
    assert latest_pull("toy_source") is None


def test_ingest_directory_copies_and_writes_source_card(
    tmp_data_dir, tmp_path, make_image, make_spec
) -> None:
    artifact = tmp_path / "download"
    make_image(artifact / "Ripe" / "a.png")
    spec = make_spec(doi="10.1234/toy", citation="Doe et al. 2026")

    pull = ingest(spec, artifact, date(2026, 11, 2))

    assert pull == tmp_data_dir / "bronze" / "toy_source" / "2026-11-02"
    assert (pull / "Ripe" / "a.png").is_file()
    card = (pull / "SOURCE.md").read_text()
    for expected in (
        "Toy source",
        "https://example.org/toy",
        "CC-BY-4.0",
        "10.1234/toy",
        "Doe et al.",
    ):
        assert expected in card
    assert (artifact / "Ripe" / "a.png").is_file()  # original left untouched


def test_ingest_unpacks_zip(tmp_data_dir, tmp_path, make_image, make_spec) -> None:
    image = make_image(tmp_path / "src" / "Ripe" / "a.png")
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(image, "repo-main/Ripe/a.png")

    pull = ingest(make_spec(), archive, date(2026, 11, 2))

    assert (pull / "repo-main" / "Ripe" / "a.png").is_file()
    assert (pull / "SOURCE.md").is_file()


def test_ingest_never_overwrites_a_pull(tmp_data_dir, tmp_path, make_image, make_spec) -> None:
    artifact = make_image(tmp_path / "download" / "Ripe" / "a.png").parent.parent
    ingest(make_spec(), artifact, date(2026, 11, 2))
    with pytest.raises(FileExistsError):
        ingest(make_spec(), artifact, date(2026, 11, 2))


def test_ingest_rejects_missing_artifact(tmp_data_dir, tmp_path, make_spec) -> None:
    with pytest.raises(FileNotFoundError):
        ingest(make_spec(), tmp_path / "nope.zip", date(2026, 11, 2))


def test_source_card_flags_non_commercial(make_spec) -> None:
    card = render_source_card(make_spec(commercial_use=False), date(2026, 11, 2), "toy.zip")
    assert "non-commercial" in card.lower()
    assert "2026-11-02" in card
    assert "toy.zip" in card


def test_ingest_unpacks_tar(tmp_data_dir, tmp_path, make_image, make_spec) -> None:
    import tarfile

    image = make_image(tmp_path / "src" / "Ripe" / "a.png")
    archive = tmp_path / "toy.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(image, "data/Ripe/a.png")

    pull = ingest(make_spec(), archive, date(2026, 11, 2))

    assert (pull / "data" / "Ripe" / "a.png").is_file()
