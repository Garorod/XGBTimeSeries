#%%
# DATA LOADING
import pandas as pd
import os
import matplotlib.pyplot as plt

DIR_PATH = '/DATA-0/home/campus.berkeley.edu/paul.digard/mfe-afp-grp13/'
os.chdir(DIR_PATH)

df = pd.read_pickle('raw_data/ep_split_ca_1').iloc[:1000000]
features = list(df)

#%%
# BUCKET FUNCTION
def bucket_data(data, y, x, bucket_size, min, max):
    import numpy as np

    cpn_diff = np.linspace(min, max, int((max-min)/bucket_size))
    avg_per_bucket = []

    for i in range(1, len(cpn_diff)):
        df_bucket = data[(data[x] < cpn_diff[i]) & (data[x] >= cpn_diff[i-1])]
        avg = df_bucket[y].mean()
        avg_per_bucket.append(avg)

    return cpn_diff, np.array(avg_per_bucket)


def plot_bucket_data(data, y, x, bucket_size, min, max, groupby_fico=False):
    if not groupby_fico:
        x_bucket, avg_y_per_bucket = bucket_data(data=data,
                                                 y=y,
                                                 x=x,
                                                 bucket_size=bucket_size,
                                                 min=min, max=max)

        fig, ax = plt.subplots()
        ax.scatter(x_bucket[1:], avg_y_per_bucket)
        ax.set_xlabel(x)
        ax.set_ylabel(f'Empirical {y} Rate')
        fig.tight_layout()
        plt.show()
    else:
        x_bucket_lt_670, avg_y_per_bucket_lt_670 = bucket_data(data=data[data['mcd_creditscore'] < 670],
                                                               y=y,
                                                               x=x,
                                                               bucket_size=bucket_size,
                                                               min=min, max=max)
        x_bucket_gt_730, avg_y_per_bucket_gt_730 = bucket_data(data=data[data['mcd_creditscore'] > 730],
                                                               y=y,
                                                               x=x,
                                                               bucket_size=bucket_size,
                                                               min=min, max=max)

        fig, ax = plt.subplots()
        ax.scatter(x_bucket_lt_670[1:], avg_y_per_bucket_lt_670, label='FICO < 670')
        ax.scatter(x_bucket_gt_730[1:], avg_y_per_bucket_gt_730, label='FICO > 730')
        ax.legend()
        ax.set_xlabel(x)
        ax.set_ylabel(f'Empirical {y} Rate')
        fig.tight_layout()
        plt.show()

#%%
# CPN DIFF VS. PREP or DFLT
x, y = 'coupon_differential', 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y, x=x,
                 bucket_size=0.005,
                 min=min, max=max,
                 groupby_fico=False)

#%%
# CPN DIFF VS. PREP or DFLT GP BY FICO
x, y = 'coupon_differential', 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y, x=x,
                 bucket_size=0.005,
                 min=min, max=max,
                 groupby_fico=True)

#%%
# CREDIT SCORE VS. PREP OR DFLT
x, y = 'mcd_creditscore', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=10,
                 min=min, max=max,
                 groupby_fico=False)

#%%
# HOUSE PRICE CHANGE VS PREP OR DFLT
df['price_change'] = df['current_property_value'] / df['mcd_originalpropertyvalue']  # / df['mcd_originalpropertyvalue']
x, y = 'price_change', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=False)

#%%
# HOUSE PRICE CHANGE VS PREP OR DFLT GP BY CREDIT SCORE
df['price_change'] = df['current_property_value'] / df['mcd_originalpropertyvalue']  # / df['mcd_originalpropertyvalue']
x, y = 'price_change', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=True)

#%%
# N OF PREVIOUS DELINQUENCIES VS. PREP OR DFLT
x, y = 'delinq_propensity', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=False)

#%%
# N OF PREVIOUS DELINQUENCIES VS. PREP OR DFLT GP BY CREDIT SCORE
x, y = 'delinq_propensity', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=True)

#%%
# LTV AT ORIGINATION VS. PREP OR DFLT
x, y = 'mcd_orig_ltv', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=False)

#%%
# LTV AT ORIGINATION VS PREP OR DFLT GP BY CREDIT SCORE
x, y = 'mcd_orig_ltv', 'default'  # 'prepayment'
min = df[x].min()
max = df[x].max()
plot_bucket_data(data=df,
                 y=y,
                 x=x,
                 bucket_size=0.05,
                 min=min, max=max,
                 groupby_fico=True)
