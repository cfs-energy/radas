# ruff: noqa: D103
"""Check that truncated or failed downloads are caught rather than silently kept."""

import urllib.request
from pathlib import Path

import pytest

from radas.adas_interface.download_adas_datasets import download_species_data, is_complete

DASHES = "C" + "-" * 70
GOOD = "\n".join(
    [
        "    1    2    2    1    1   /ARGON /TEST ADF11",
        "-" * 70,
        "  1.00000  2.00000",
        "  1.00000  2.00000",
        "-" * 20 + "/ IPRT= 1  / IGRD= 1  / Z1= 1",
        " -1.00000 -2.00000 -3.00000 -4.00000",
        DASHES,
        "C  Comment footer written by ADAS",
        DASHES,
        "",
        DASHES,
        "C This file has been downloaded from OPEN-ADAS (http://open.adas.ac.uk)",
        "C Terms and conditions",
        DASHES,
        "",
    ]
)
ERROR_PAGE = "OPEN-ADAS Error: no such file\n"
EXPECTED_FILENAME = "argon_effective_ionisation_89.dat"


def serve(monkeypatch, *responses):
    """Replace urlretrieve: each call writes the next response, then raises it if it is an exception (a dropped connection)."""
    queue = list(responses)

    def urlretrieve(url, filename):
        response = queue.pop(0)
        Path(filename).write_text(str(response))
        if isinstance(response, Exception):
            raise response

    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)


def download(tmp_path):
    species_config = {"atomic_symbol": "Ar", "data_files": {"effective_ionisation": 89}}
    data_file_config = {"adf11": {"effective_ionisation": {"prefix": "SCD"}}}
    download_species_data(tmp_path, "argon", species_config, data_file_config, verbose=0)
    return tmp_path / EXPECTED_FILENAME


def test_only_the_whole_file_is_complete(tmp_path):
    f = tmp_path / "a.dat"
    f.write_text(GOOD)
    assert is_complete(f)

    for n in range(len(GOOD) - len(DASHES)):  # every truncation that cuts off the final divider line, including the empty file
        f.write_text(GOOD[:n])
        assert not is_complete(f), f"accepted a file truncated to {n} characters"


def test_complete_download_is_kept_and_then_reused(tmp_path, monkeypatch):
    serve(monkeypatch, GOOD)
    assert download(tmp_path).read_text() == GOOD
    assert download(tmp_path).read_text() == GOOD  # the fake server has no second response, so this must not download


def test_truncated_download_raises_and_is_removed(tmp_path, monkeypatch):
    serve(monkeypatch, GOOD[: len(GOOD) // 2])
    with pytest.raises(RuntimeError, match="incomplete"):
        download(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_is_replaced_on_the_next_run(tmp_path, monkeypatch):
    serve(monkeypatch, ConnectionError(GOOD[:100]), GOOD)  # writes a partial file, then raises
    with pytest.raises(ConnectionError):
        download(tmp_path)
    assert download(tmp_path).read_text() == GOOD


def test_open_adas_error_is_removed_without_raising(tmp_path, monkeypatch):
    serve(monkeypatch, ERROR_PAGE)
    assert not download(tmp_path).exists()
