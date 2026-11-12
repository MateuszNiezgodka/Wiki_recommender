import zipfile

import pytest

from wiki_recommender import download


def _fake_data(root):
    for name in download.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def test_pack_and_unpack_restore_every_file(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _fake_data(source)
    archive = download.pack(source, tmp_path / "data.zip")
    written = download.unpack(archive, target)
    assert len(written) == len(download.FILES)
    assert all((target / name).read_text() == name for name in download.FILES)


def test_pack_names_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="recommender.pkl"):
        _fake_data(tmp_path)
        (tmp_path / "data/processed/recommender.pkl").unlink()
        download.pack(tmp_path, tmp_path / "data.zip")


def test_unpack_refuses_unexpected_entries(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "nope")
    with pytest.raises(ValueError, match="unexpected"):
        download.unpack(archive, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()
