"""Area selection: full extent vs. bounding-box region."""
import pandas as pd

from cartostemma.collation import select_loci


def _loci():
    # four loci on a line; x in {0, 100, 200, 300}
    return pd.DataFrame(
        {"x": [0.0, 100.0, 200.0, 300.0], "y": [0.0, 0.0, 0.0, 0.0]},
        index=["0_0", "0_1", "0_2", "0_3"],
    )


def test_full_extent():
    loci = _loci()
    assert set(select_loci(loci, area=None)) == set(loci.index)


def test_bounding_box():
    loci = _loci()
    # box around x in [90, 210] keeps the two middle loci
    keep = select_loci(loci, area=(90, -10, 210, 10))
    assert set(keep) == {"0_1", "0_2"}


def test_bounding_box_with_buffer():
    loci = _loci()
    # a tight box plus a 15 m buffer pulls in the neighbour at x=100
    keep = select_loci(loci, area=(0, -1, 90, 1), buffer=15.0)
    assert set(keep) == {"0_0", "0_1"}


if __name__ == "__main__":
    test_full_extent()
    test_bounding_box()
    test_bounding_box_with_buffer()
    print("ok")
