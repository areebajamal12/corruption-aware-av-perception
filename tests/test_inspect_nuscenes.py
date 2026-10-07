from pathlib import Path

import pytest

from av_perception.data.inspect_nuscenes import (
    category_is_relevant,
    resolve_sample,
    validate_layout,
)


class FakeNuScenes:
    def __init__(self) -> None:
        self.sample = [{"token": "first"}]

    def get(self, table: str, token: str) -> dict[str, str]:
        return {"table": table, "token": token}


def test_relevant_categories() -> None:
    assert category_is_relevant("vehicle.car", ("vehicle.", "human.pedestrian."))
    assert category_is_relevant("human.pedestrian.adult", ("vehicle.", "human.pedestrian."))
    assert not category_is_relevant("movable_object.barrier", ("vehicle.",))


def test_resolve_sample_defaults_to_first() -> None:
    assert resolve_sample(FakeNuScenes(), None) == {"token": "first"}


def test_resolve_sample_uses_requested_token() -> None:
    assert resolve_sample(FakeNuScenes(), "chosen") == {
        "table": "sample",
        "token": "chosen",
    }


def test_validate_layout_reports_missing_folders(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="samples"):
        validate_layout(tmp_path, "v1.0-mini")


def test_validate_layout_accepts_expected_folders(tmp_path: Path) -> None:
    (tmp_path / "samples").mkdir()
    (tmp_path / "v1.0-mini").mkdir()
    validate_layout(tmp_path, "v1.0-mini")
