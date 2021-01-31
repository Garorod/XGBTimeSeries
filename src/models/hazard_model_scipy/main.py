import pandas as pd
import os
import matplotlib.pyplot as plt
from functions_likelihood import get_optimal_params

DIR_PATH = '/DATA-0/home/campus.berkeley.edu/paul.digard/mfe-afp-grp13/'
os.chdir(DIR_PATH)

df = pd.read_pickle('raw_data/ep_split_ca_1').iloc[:100]
features = list(df)
x_names = ['mcd_creditscore', 'mcd_orig_ltv', 'mcd_orig_term', 'coupon_differential']
df['price_change'] = df['current_property_value'] / df['mcd_originalpropertyvalue']  # / df['mcd_originalpropertyvalue']
df['period_begin'] = df['month_count']
df['period_end'] = df['month_count'] + 1
df_uncens = df[df['prepayment'] == 1]

params, success = get_optimal_params(x_names, df_uncens, df)
