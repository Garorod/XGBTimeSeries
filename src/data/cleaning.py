# %%
import pandas as pd
from datetime import date
from os.path import split as split_path
from tqdm import tqdm
from joblib import Parallel, delayed

from src.data import trimmed_data_dir
from src.data import parquet_paths
from src.data.constants import PREPAY_STATUSES, DEFAULT_STATUSES, x_cols, y_cols


# %%
def trim_to_first_60_day_delinq(raw_df):
    raw_df = raw_df.drop(['current_date', 'current_month', 'current_year'], axis=1)
    current_month = (raw_df['orig_month'] + raw_df['month_count'] -1 ) % 12 + 1
    current_year = raw_df['orig_year'] + (raw_df['orig_month'] + raw_df['month_count'] -1) // 12
    current_date = current_year.apply(str) + '-' + current_month.apply('{:02}'.format) + '-01'
    current_date = pd.to_datetime(current_date)

    # The current date (year) in the raw data in wrong
    raw_df = raw_df.assign(current_date=current_date, current_year=current_year, current_month=current_month)

    post_crisis = (raw_df[raw_df['sr_date_transfer'] >= 20090101])
    post_crisis = post_crisis.set_index(['mcd_loanid', 'sr_unique_id', 'sr_property_id', pd.to_datetime(post_crisis['current_date'])]).sort_index()

    # Regard 60-day as absorbing state, and ignore entries with unknown status
    delinq = post_crisis['payment_current_status'].isin(list(DEFAULT_STATUSES))
    first_delinquent_date = (delinq[delinq]
                .reset_index('current_date')
                .groupby('mcd_loanid')['current_date']
                .first()
                .rename('first_delinquent_date'))

    post_crisis['first_delinquent_date'] = first_delinquent_date.reindex(delinq.index, level='mcd_loanid')

    no_delinq = post_crisis['first_delinquent_date'].isnull()
    before_deliq = post_crisis['current_date'] <= post_crisis['first_delinquent_date']
    missing_status = post_crisis['payment_current_status'] == '-'
    trimmed_df = post_crisis[(no_delinq|before_deliq) & ~missing_status].drop('first_delinquent_date', axis=1)
    
    # Categories to predict
    X = trimmed_df.reindex(x_cols, axis=1)
    X['prepaid'] = trimmed_df['payment_current_status'].isin(list(PREPAY_STATUSES)).astype(int)
    X['default'] = trimmed_df['payment_current_status'].isin(list(DEFAULT_STATUSES)).astype(int)
    X['current'] = 1 - (X['prepaid'] | X['default'])
    return X


#%%
def trim_and_save(file_path):
    file_name = split_path(file_path)[-1]
    df = pd.read_parquet(file_path)
    t_df = trim_to_first_60_day_delinq(df)
    loanid = t_df.index.get_level_values('mcd_loanid')
    hashkey = pd.util.hash_array(loanid) % 100
    for key, g_df in t_df.groupby(hashkey):
        g_path = trimmed_data_dir + file_name + '.bucket_' + f'{key:02}'
        g_df.to_parquet(g_path)
    return file_name
#%%
if __name__ == '__main__':
    Parallel(n_jobs=8)(delayed(trim_and_save)(p) for p in tqdm(parquet_paths))

# %%
# # %% CHECK if dates are correct
# g = post_crisis.groupby('mcd_loanid')
# mono = g['month_count'].is_monotonic_increasing
# mono