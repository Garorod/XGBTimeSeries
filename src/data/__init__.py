from glob import glob

from src.data.constants import *
from src.data.load import debug_buckets, train_buckets, valid_buckets, test_buckets
parquet_paths = glob(parquet_dir + '*')