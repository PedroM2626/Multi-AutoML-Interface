"""Guards around what load_data accepts as a dataset.

data_lake/raw holds files committed through LFS that were never pulled; pandas read them as a
one-column CSV of pointer text and the Training page then offered that column as the date column.
The image-directory branch has a second shape: a folder of images plus the annotations CSV the CV
upload stores inside it, which is what Computer Vision Multi-Label Classification trains on.
"""
import pandas as pd
import pytest

from src.data_utils import CV_IMAGE_COLUMN, cv_label_columns, load_data, validate_cv_annotations

LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:f454a2fc46d8e6de382ff3bef1a0b0b2c8f0d9e4a1c2b3d4e5f60718293a4b5c\n"
    b"size 1024\n"
)


def test_git_lfs_pointer_is_rejected_with_a_readable_message(tmp_path):
    stub = tmp_path / "reflex_ui_dataset.csv"
    stub.write_bytes(LFS_POINTER)

    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_data(str(stub))


def test_a_real_csv_still_loads(tmp_path):
    real = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]}).to_csv(real, index=False)

    df = load_data(str(real))

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def _image_folder(tmp_path):
    folder = tmp_path / "images"
    (folder / "red").mkdir(parents=True)
    (folder / "red" / "shot.png").write_bytes(b"\x89PNG")
    return folder


def test_an_annotated_image_folder_loads_as_a_table(tmp_path):
    folder = _image_folder(tmp_path)
    pd.DataFrame({CV_IMAGE_COLUMN: ["red/shot.png"], "fog": [1], "ripe": [0]}).to_csv(
        folder / "annotations.csv", index=False
    )

    frame = load_data(str(folder))

    assert cv_label_columns(frame.columns) == ["fog", "ripe"]
    assert frame["Image_Directory"].iloc[0] == str(folder)


def test_a_folder_without_annotations_keeps_the_directory_stub(tmp_path):
    folder = _image_folder(tmp_path)

    frame = load_data(str(folder))

    assert list(frame.columns) == ["Image_Directory", "Total_Images", "Type"]


def test_an_annotation_table_needs_the_image_column_and_two_labels():
    with pytest.raises(ValueError, match="image"):
        validate_cv_annotations(pd.DataFrame({"fog": [1], "ripe": [0]}))
    with pytest.raises(ValueError, match="at least two label columns"):
        validate_cv_annotations(pd.DataFrame({CV_IMAGE_COLUMN: ["a.png"], "fog": [1]}))
