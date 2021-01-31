# %%
import pandas as pd
from glob import glob
import os.path
from joblib import Parallel, delayed
from tqdm import tqdm

from imblearn.over_sampling import RandomOverSampler

from src.data import trimmed_data_dir, x_cols, y_cols


def make_bucket_sep_index(bucket, return_index=False, oversample=True):
    bucket_paths = glob(os.path.join(trimmed_data_dir, f'*.bucket_{bucket:02}'))
    bucket_data = pd.concat(pd.read_parquet(p) for p in bucket_paths)
    # loan id number too big for joblib
    val = bucket_data.reset_index(drop=True)
    X, y = val.reindex(x_cols, axis=1), val.reindex(y_cols, axis=1)
    if oversample:
        X, y = RandomOverSampler().fit_resample(X.values, y.values)
        X = pd.DataFrame(X, columns=x_cols)
        y = pd.DataFrame(y, columns=y_cols)
    if return_index:
        idx = bucket_data.index.to_frame().reset_index(drop=True).astype(str)
        return X, y, idx
    else:
        return X, y


# %%
def make_buckets(buckets, return_index=False, oversample=True):
    xys = Parallel(n_jobs=8)(delayed(make_bucket_sep_index)(buc, return_index, oversample) for buc in tqdm(buckets))
    idxs, Xs, ys = [], [], []
    for xy in xys:
        if return_index:
            X, y, idx = xy
            idxs.append(idx)
        else:
            X, y = xy
        Xs.append(X)
        ys.append(y)
    X, y = pd.concat(Xs), pd.concat(ys)
    if return_index:
        return X, y, pd.concat(idxs)
    else:
        return X, y

# %%
debug_bucket = range(2)
train_bucket = range(80)
valid_bucket = range(80, 90)
test_bucket = range(90, 100)

def debug_buckets(): return make_buckets(debug_bucket)
def train_buckets(): return make_buckets(train_bucket)
def valid_buckets(): return make_buckets(valid_bucket)
def test_buckets(): return make_buckets(test_bucket)
# %%

