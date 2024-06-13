from scipy.stats import bootstrap as retarded_bootstrap
import krippendorff
import pandas as pd
import numpy as np



def chunks(lst, n):
    """Return successive n-sized chunks from lst."""
    outs = []
    for i in range(0, len(lst), n):
        outs.append(lst[i:i + n])
    return outs

#########################
## AGREEMENT
#########################

class stat:
    fleiss_kappa = "Fleiss' kappa"
    kripp_alpha = "Krippendorff's alpha"
    kend_tau = "Kendall's tau"
    mcfad_r2 = "McFadden's pseudo-R-squared"
    r2 = "R-Squared"
    ci_low = "CI low"
    ci_high = "CI high"
    proportion = 'proportion'
    mean = 'mean'
    n = 'n'
    likert_dialogue_quality = 'likert dialogue quality'
    likert_turn_quality = 'likert turn quality'
    p_of_f_test = 'P value of F-test'
    p_of_llr_test = 'P value of LLR-test'
    likert = 'likert'

def bootstrap_ci(data, statistic_fn, n_resamples=10**3, confidence_level=0.95):
    wrapped_data = [dict(point=d) for d in data]
    statistic_fn_wrapper = lambda ds: statistic_fn([d['point'] for d in ds])
    result = retarded_bootstrap((wrapped_data,), statistic_fn_wrapper, vectorized=False,
                                n_resamples=n_resamples, confidence_level=confidence_level)
    return result.confidence_interval

def krippendorffs_alpha(df, ci=True, to_string=False, level_of_measurement='ordinal'):
    """
    :param df: pandas dataframe: items x labeler: label
    :return:
    """
    ratings = df.to_numpy()
    if to_string:
        ratings = ratings.astype('U')
    ka = lambda x: krippendorff.alpha(x.T, level_of_measurement=level_of_measurement)
    try:
        alpha = ka(ratings)
    except AssertionError:
        alpha = None
    if ci:
        try:
            low, high = bootstrap_ci(ratings, lambda x: ka(np.array(x)))
        except AssertionError:
            low, high = None, None
        result = {
            stat.kripp_alpha: alpha,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.kripp_alpha: alpha,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)