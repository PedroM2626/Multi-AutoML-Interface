"""The data loader must not accept a Git LFS pointer as a dataset.

data_lake/raw holds files committed through LFS that were never pulled; pandas read them as a
one-column CSV of pointer text and the Training page then offered that column as the date column.
"""
import pandas as pd
import pytest

from src.data_utils import load_data

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
