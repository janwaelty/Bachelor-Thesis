"""
Computation of various metrics concerning the stability of the estimators

Author: Jan Wälty 2023

"""

from gerber import gerber_cov_stat1 as gerber
from cov1para import cov1Para
from CovCor import covCor
import numpy as np
from statistics import mean
from pyfinance import TSeries
import pandas as pd


def calc_frobenius_norm(sample, population) :
    # return np.linalg.norm(population - sample, ord='fro') # MAE-like
    return np.linalg.norm(population - sample, ord='fro') ** 2  # MSE-like


def pop_cov_return(data) :
    sample = data.dropna()
    return sample.cov().to_numpy()


def get_frob(data, win_length, method, max_length=120) :
    """
    :param data:
    :param win_length: either 24, 60 or 120
    :param method: GS1, SM, SM2, HC
    :param max_length:
    :return: average Frobenius norm between "true" covariance matrix and estimated
    """

    norm_list = []
    true_cov_mat_span = data.iloc[max_length - win_length :, :]
    if method == "GS1" :
        for idx in range(len(data) - max_length) :
            sample = data.iloc[idx + max_length - win_length :idx + max_length, :]
            gerber_sample, _ = gerber(sample.values)
            norm_list.append(calc_frobenius_norm(gerber_sample, pop_cov_return(true_cov_mat_span)))
    elif method == "SM" :
        for idx in range(len(data) - max_length) :
            sample = data.iloc[idx + max_length - win_length :idx + max_length, :]
            sm_sample, _ = covCor(sample.values)
            norm_list.append(calc_frobenius_norm(sm_sample, pop_cov_return(true_cov_mat_span)))
    elif method == "SM2" :
        for idx in range(len(data) - max_length) :
            sample = data.iloc[idx + max_length - win_length :idx + max_length, :]
            sm2_sample, _ = cov1Para(sample.values)
            norm_list.append(calc_frobenius_norm(sm2_sample, pop_cov_return(true_cov_mat_span)))
    else :
        for idx in range(len(data) - max_length) :
            sample = data.iloc[idx + max_length - win_length :idx + max_length, :]
            hc_sample = sample.cov().to_numpy()
            norm_list.append(calc_frobenius_norm(hc_sample, pop_cov_return(true_cov_mat_span)))
    return norm_list

def frob_df(data, win_length_list) :
    methods = ["GS1", "SM", "SM2", "HC"]
    df = pd.DataFrame()
    df["Methods"] = methods
    for win_length in win_length_list :
        entry_list = []
        for method in methods :
            entry_list.append(round(mean(get_frob(data, win_length, method)), 10))
        df[win_length] = entry_list
    return df


def frob_norm(data, win_length, method, constant = 0.5, max_length=120) :
    """
    calculates the frobenius norm series of an estimator
    :param data:
    :param win_length:
    :param method:
    :param constant: gerber constant
    :param max_length: 120 per default
    :return: list of Frobenius norms
    """
    frob_norm = []
    for idx in range(len(data) - max_length) :
        sample = data.iloc[idx + max_length - win_length :idx + max_length, :]
        if method == "HC" :
            cov_mat = sample.cov().to_numpy()
        elif method == "SM" :
            cov_mat, _ = covCor(sample.values)
        elif method == "SM2" :
            cov_mat, _ = cov1Para(sample.values)
        else :
            cov_mat, _ = gerber(sample.values, constant)
        frob_norm.append(np.linalg.norm(cov_mat, ord='fro'))
    return frob_norm




if __name__ == "__main__" :
    file_path = "C:\\Universität\\Numerical Methods\\prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['date'], index_col=["date"]).pct_change().dropna()
    lookback_window_list = [24, 60, 120]

    # frob_df(rets_df, lookback_window_list).to_csv("Frobenius_norm_squared.csv", index=False)

    """
    First difference Frobenius norm 
    """
    methods = ["HC", "GS", "SM", "SM2"]
    win_lenghts = [24,60,120]
    df = pd.DataFrame()
    for length in win_lenghts:
        for method in methods :
            df[method] = frob_norm(rets_df, length, method)
       # df.to_csv("frobenius_time_series_%d.csv" % length, index=False)
        df_diff = pd.DataFrame()
        for method in methods :
            df_diff[method] = df[method].diff().dropna()
      #  df_diff.to_csv("frobenius_time_series_first_diff_%d.csv" % length, index=False)



    """
    Frobenius norm standard deviation for Gerber statistics for different Gerber constant 
    """
    gerber_frobenius = pd.DataFrame()
    gerber_frobenius["Constant"] = [x for x in np.arange(0, 1.1, 0.1)]
    for size in [24,60,120]:
        stddev_list = []
        for constant in np.arange(0, 1.1, 0.1):
            entry= frob_norm(rets_df, size, "GS", constant)
            stddev_list.append(np.std(entry))
        gerber_frobenius[size] = stddev_list


    gerber_frobenius.to_csv("Frobenius_norm_gerber.csv", index = False)
