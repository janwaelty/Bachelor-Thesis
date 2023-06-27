"""
Author: Jan Wälty

Compute performance metrics from files with portfolio values
- annualized arithmetic return
- annualized standard deviation of returns
- annualized Sharpe ratio (rf = 3 month US-Treasury Yield)

"""

from pyfinance.datasets import load_rf
import pandas as pd
from pyfinance import TSeries


def get_percentage_change(file_path) :
    return pd.read_csv(file_path, parse_dates=['date'], index_col=["date"]).pct_change().dropna()


def get_risk_free(start) :
    rf = pd.DataFrame(load_rf(freq="M"), columns=["rate"])
    rf.index = pd.to_datetime(rf.index)
    mask = (rf.index > start) & (rf.index <= end_date)
    rf = rf.loc[mask]
    return rf


def asset_stdev(data) :
    sd_list = []
    for idx in range(data.shape[1]) :
        ret = data.iloc[:, idx]
        dat = TSeries(ret, freq="M")
        sd_list.append(round(100 * dat.std() * (12 ** 0.5), 2))
    return sd_list


def asset_returns(data) :
    ret_list = []
    for idx in range(data.shape[1]) :
        ret = data.iloc[:, idx]
        dat = TSeries(ret, freq="M")
        annual_rets = dat.rollup('A')
        # ret_list.append(round(100 * dat.mean() * 12, 2))
        ret_list.append(round(100 * annual_rets.mean(), 2))
    return ret_list


def asset_sharpe(data) :
    sharpe_list = []
    for idx in range(data.shape[1]) :
        rf = get_risk_free(start_date_asset)
        ret = data.iloc[:, idx].dropna()
        ret = ret.to_numpy()
        rf["ret"] = ret.tolist()
        rf["excess"] = rf["ret"] - rf["rate"]
        monthly_excess_ret = rf["excess"].mean()
        sd_excess = rf["excess"].std()
        sharpe_list.append(round(monthly_excess_ret / sd_excess * (12 ** 0.5), 2))
    return sharpe_list


def asset_df(data) :
    df = pd.DataFrame()
    df["Asset"] = ["SPX", "RTY", "MXEA", "MXEF", "XAU", "SPGSCI", "LF98TRUU", "LBUSTRUU", "FNERTR"]
    df["Average annualized return"] = asset_returns(data)
    df["Annualized Sharpe Ratio"] = asset_sharpe(data)
    df["Annualized standard deviation"] = asset_stdev(data)
    return df


def get_annualized_sharpe(data) :
    sharpe_list = []
    #idx_levels = [1, 4, 7, 10, 13]
    idx_levels = [15]
    for idx in idx_levels :
        rf = get_risk_free(start_date)
        ret = data.iloc[:, idx].dropna()
        ret = ret.to_numpy()
        rf["ret"] = ret.tolist()
        rf["excess"] = rf["ret"] - rf["rate"]
        monthly_excess_ret = rf["excess"].mean()
        sd_excess = rf["excess"].std()
        sharpe_list.append(round(monthly_excess_ret / sd_excess * (12 ** 0.5), 2))
    return sharpe_list


def get_annualized_sdev(data) :
    sd_list = []
    idx_levels = [1, 4, 7, 10, 13]
    #idx_levels = [15] # global minimum variance portfolio
    for idx in idx_levels:
        ret = data.iloc[:, idx]
        dat = TSeries(ret, freq="M")
        sd_list.append(round(100 * dat.std() * (12 ** 0.5), 2))
    return sd_list


def get_annualized_turnover(data) :
    turnover_list = []
    idx_level = [1, 4, 7, 10, 13]
    for idx in idx_level :
        annualized_turnover = 12 * data.iloc[1:, idx].mean()  # first entry dropped
        turnover_list.append(round(annualized_turnover,2))
    return turnover_list


def get_turnover_df(methods, win_lengths) :
    counter = 0
    levels = [3, 6, 9, 12, 15]
    df = pd.DataFrame()
    df["Level"] = levels
    for win_length in win_lengths :
        for method in methods :
            turnover_data = pd.read_csv("C:\\Universität\\Numerical Methods\\%dyr_threshold0.50_%s_turnover.csv"
                                        % (win_length, method), parse_dates=['date'], index_col=['date']).dropna()
            df[counter] = get_annualized_turnover(turnover_data)
            counter += 1
    return df


def get_annualized_return(data) :
    ret_list = []
    #idx_levels = [1, 4, 7, 10, 13]
    idx_levels = [15]
    for idx in idx_levels :
        ret = data.iloc[:, idx]
        dat = TSeries(ret, freq="M")
        ret_list.append(round(100 * dat.mean()*12, 2))
    return ret_list


def generate_df_estimator(method, measure) :
    """
    computes metrics for specific estimator
    :param method:
    :param measure:
    :return:
    """
    levels = [3, 6, 9, 12, 15]
    df = pd.DataFrame()
    df["Level"] = levels
    if measure == "ret" :  # annualized arithmetic return
        for win_len in method :
            ret_list = pd.read_csv(win_len, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            ret_list_method = get_annualized_return(ret_list)
            df[method[win_len]] = ret_list_method
    elif measure == "sd" :  # annualized standard deviation
        for win_len in method :
            sd_list = pd.read_csv(win_len, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            sd_list_method = get_annualized_sdev(sd_list)
            df[method[win_len]] = sd_list_method
    elif measure == "sh" :  # annualized Sharpe ratio
        for win_len in method :
            sharpe_list = pd.read_csv(win_len, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            sharpe_list_method = get_annualized_sharpe(sharpe_list)
            df[method[win_len]] = sharpe_list_method

    return df


def generate_df(methods, measure) :
    """
    computes metrics for all methods
    :param methods: GS1, SM, SM2 or HC
    :param measure:
    :return:
    """
    levels = [3, 6, 9, 12, 15]
    df = pd.DataFrame()
    df["Level"] = levels
    if measure == "ret" :  # annualized geometric return
        for method in methods :
            ret_list = pd.read_csv(method, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            ret_list_method = get_annualized_return(ret_list)
            df[methods[method]] = ret_list_method
    elif measure == "sd" :  # annualized standard deviation
        for method in methods :
            sd_list = pd.read_csv(method, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            sd_list_method = get_annualized_sdev(sd_list)
            df[methods[method]] = sd_list_method
    elif measure == "sh" :  # annualized Sharpe ratio
        for method in methods :
            sharpe_list = pd.read_csv(method, parse_dates=['date'], index_col=['date']).pct_change().dropna()
            sharpe_list_method = get_annualized_sharpe(sharpe_list)
            df[methods[method]] = sharpe_list_method

    return df


if __name__ == "__main__" :
    """
    Configuration of input values
    - lookback window
    - threshold for Gerber statistics
    - start date (begin of performance measurement)
        - used for risk free rate
    - end date (end of performance measurement) 
       - used for risk free rate 
    """
    lookback_window = 10
    gs_threshold = 0.5
    end_date = "2020-12-31"
    start_date = "1998-02-01"
    start_date_asset = "1998-02-01"

    """""""""""""""""""""""""""""""""""""""""""""""""""
    generate files for comparison of estimators
    """""""""""""""""""""""""""""""""""""""""""""""""""
    filename = {"C:\\Universität\\Numerical Methods\\without_%dyr_threshold%.2f_HC_value.csv" %
                (lookback_window, gs_threshold) : "HC",
                "C:\\Universität\\Numerical Methods\\without_%dyr_threshold%.2f_SM_value.csv" %
                (lookback_window, gs_threshold) : "SM",
                "C:\\Universität\\Numerical Methods\\without_%dyr_threshold%.2f_SM2_value.csv" %
                (lookback_window, gs_threshold) : "SM2",
                "C:\\Universität\\Numerical Methods\\without_%dyr_threshold%.2f_GS1_value.csv" %
                (lookback_window, gs_threshold) : "GS1"
                }

    generate_df(filename, "ret").to_csv("minvar_without_%dyr_threshold%.1f_return.csv" %
                                        (lookback_window, gs_threshold), index=False)
    generate_df(filename, "sd").to_csv("minvar_without_%dyr_threshold%.1f_sd.csv" %
                                       (lookback_window, gs_threshold), index=False)
    generate_df(filename, "sh").to_csv("minvar_without_%dyr_threshold%.1f_sharpe.csv" %
                                       (lookback_window, gs_threshold), index=False)

    """""""""""""""""""""""""""""""""""""""""""""""""""
    generate files for comparison of window lengths
    """""""""""""""""""""""""""""""""""""""""""""""""""

    methods = ["GS1", "SM", "SM2", "HC"]
    for method in methods :
        filename = {"C:\\Universität\\Numerical Methods\\without_2yr_threshold%.2f_%s_value.csv" %
                    (gs_threshold, method) : "2",
                    "C:\\Universität\\Numerical Methods\\without_5yr_threshold%.2f_%s_value.csv" %
                    (gs_threshold, method) : "5",
                    "C:\\Universität\\Numerical Methods\\without_10yr_threshold%.2f_%s_value.csv" %
                    (gs_threshold, method) : "10"}

        generate_df_estimator(filename, "ret").to_csv("minvar_without_threshold%.1f_%s_return.csv" %
                                                      (gs_threshold, method), index=False)
        generate_df_estimator(filename, "sd").to_csv("minvar_without_threshold%.1f_%s_sd.csv" %
                                                     (gs_threshold, method), index=False)
        generate_df_estimator(filename, "sh").to_csv("minvar_without_threshold%.1f_%s_sharpe.csv" %
                                                     (gs_threshold, method), index=False)

    #get_percentage_change("C:\\Universität\\Numerical Methods\\prcs.csv").to_csv("prcs_change.csv")

    """""""""""""""""""""""""""""""""""""""""""""""
    turnover data
    """""""""""""""""""""""""""""""""""""""""""""""

    methods = ["GS1", "SM", "SM2", "HC"]
    lengths = [2, 5, 10]
    #get_turnover_df(methods, lengths).to_csv("test_turnover.csv")