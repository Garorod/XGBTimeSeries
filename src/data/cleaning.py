# %%
from src.data import parquet_paths
import pandas as pd
# %%
p = '/DATA-0/mfe-afp-grp13/yawer/afp/data/ep_split_ca_1.parquet'
df = pd.read_parquet(p).drop(['current_date', 'current_month', 'current_year'], axis=1)
# %%
from datetime import date
from dateutil.relativedelta import relativedelta

current_date = df[['orig_year', 'orig_month', 'month_count']].apply(
    lambda r: date(r[0], r[1], 1) + relativedelta(months=r[2]), axis=1)
current_date = pd.to_datetime(current_date)

df = df.assign(current_date=current_date, current_year=current_date.dt.year, current_month=current_date.dt.month)

#%%
post_crisis = (df[df['sr_date_transfer'] >= 20090101])
post_crisis = post_crisis.set_index(['mcd_loanid', 'sr_unique_id', 'sr_property_id', pd.to_datetime(post_crisis['current_date'])]).sort_index()


# # %% CHECK if dates are correct
# g = post_crisis.groupby('mcd_loanid')
# mono = g['month_count'].is_monotonic_increasing
# mono
# %%
delinq = post_crisis['payment_current_status'].isin(list('XRBF234'))
first_delinquent_date = (delinq[delinq]
            .reset_index('current_date')
            .groupby('mcd_loanid')['current_date']
            .first()
            .rename('first_delinquent_date'))

# post_crisis
# %%
