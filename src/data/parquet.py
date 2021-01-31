# %%
import pandas as pd
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from src.data.constants import parquet_dir, raw_data_dir

# %%
if __name__ == '__main__':
	def pickle_to_parquet(p):
		file_name = p.replace(raw_data_dir, '')
		pd.read_pickle(p).to_parquet(parquet_dir + file_name + '.parquet')
	
	paths = glob(raw_data_dir + '*')
	Parallel(n_jobs=8)(delayed(pickle_to_parquet)(p) for p in tqdm(paths))
# %%
