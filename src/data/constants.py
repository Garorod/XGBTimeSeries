parquet_dir = '/DATA-0/mfe-afp-grp13/yawer/afp/data//'
raw_data_dir = '/DATA-0/mfe-afp-grp13/raw_data/'
trimmed_data_dir = '/DATA-0/mfe-afp-grp13/yawer/afp/trimmed_data//'

PREPAY_STATUSES = 'PT'
DEFAULT_STATUSES = 'XRBF234'
CURRENT_STATUSES = 'C01'

x_cols = ['attom_site_zip', 'mcd_originalloanamount',
          'dum_gse', 'dum_ginnie',
          'mcd_creditscore', 'mcd_orig_ltv', 'mcd_orig_rate', 'mcd_orig_term',
          'mcd_originalpropertyvalue', 'orig_year', 'orig_month',
          'dum_bank', 'dum_nonbank',
          'original_house_price_index', 'month_count',
          'current_year', 'current_month',
          'current_loan_balance', 'Rate', 'house_price_index',
          'coupon_differential', 'current_property_value', 'delinq_propensity']

y_cols = ['prepaid', 'default', 'current']

