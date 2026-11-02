import numpy as np
import pandas as pd

from cascadepahs.validation import (folds_blocks, folds_lodo, folds_random, folds_station_exclusion,
                                    folds_winter_to_summer, summer_block_ids)


def _dates(df, idx):
    return set(df.iloc[idx]["date"].astype(str))


def test_primary_scheme_structure(dasha):
    folds = folds_blocks(dasha, 4)
    assert [f.name for f in folds] == [f"S{i}" for i in range(1, 9)] + ["W"]
    summer = dasha["campaign"].str.startswith("summer").to_numpy()
    tested = np.concatenate([f.test for f in folds])
    assert sorted(tested) == list(range(len(dasha)))                  # every sample tested exactly once
    for f in folds:
        assert not (_dates(dasha, f.train) & _dates(dasha, f.test))   # no shared survey date
    w = folds[-1]
    assert summer[w.train].all() and (~summer[w.test]).all()          # winter predicted from summer only
    # blocks are 4 consecutive calendar days
    for f in folds[:-1]:
        d = pd.to_datetime(dasha.iloc[f.test]["date"])
        assert (d.max() - d.min()).days <= 3


def test_buffers_and_offsets(dasha):
    f0 = folds_blocks(dasha, 4, buffer_days=2, include_winter=False)[2]
    test_d = pd.to_datetime(dasha.iloc[f0.test]["date"])
    train_d = pd.to_datetime(dasha.iloc[f0.train]["date"])
    summer_train = train_d[train_d.dt.year == 2024]
    gap = np.min(np.abs((summer_train.to_numpy()[:, None] - test_d.to_numpy()[None, :]).astype("timedelta64[D]").astype(int)))
    assert gap > 2
    ids = summer_block_ids(dasha, 4, offset=1)
    first = dasha[ids == 0]["date"]
    assert pd.to_datetime(first).nunique() <= 3


def test_other_schemes(dasha):
    assert len(folds_lodo(dasha)) == dasha["date"].nunique()
    assert len(folds_lodo(dasha, summer_only_test=True)) == 32
    (f,) = folds_winter_to_summer(dasha)
    assert dasha.iloc[f.train]["campaign"].str.startswith("winter").all()
    fr = folds_random(dasha, n_repeats=3, seed=0)
    assert len(fr) == 3 and all(abs(len(x.test) - 0.2 * len(dasha)) <= 2 for x in fr)
    for x in folds_station_exclusion(dasha):
        s = x.info["station"]
        assert (dasha.iloc[x.test]["station"] == s).all()
        assert (dasha.iloc[x.train]["station"] != s).all()
