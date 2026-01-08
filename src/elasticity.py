
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample

def estimate_loglog_ols(df_ts, price_col='avg_price', qty_col='units', controls=None):
    df = df_ts.copy().dropna(subset=[price_col, qty_col])
    df = df[df[qty_col] > 0]

    df['log_q'] = np.log(df[qty_col])
    df['log_p'] = np.log(df[price_col])

    df['promo_flag'] = (df[price_col] < df[price_col].median() * 0.95).astype(int)
    df['month'] = pd.to_datetime(df['week']).dt.month

    formula = 'log_q ~ log_p + promo_flag + C(month)'
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    beta = model.params.get('log_p', np.nan)

    boots = []
    for i in range(500):
        sample = df.sample(frac=1, replace=True)
        try:
            m = smf.ols(formula=formula, data=sample).fit()
            boots.append(m.params.get('log_p'))
        except:
            pass

    ci_lower = np.percentile(boots, 2.5) if len(boots) > 0 else np.nan
    ci_upper = np.percentile(boots, 97.5) if len(boots) > 0 else np.nan

    return {
        'model': model,
        'elasticity': beta,
        'ci': (ci_lower, ci_upper),
        'n_obs': len(df)
    }
