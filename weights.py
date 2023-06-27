import numpy as np
import pandas as pd
from portfolio_optimizer import portfolio_optimizer


def sample_mean_return(weights, returns) :
    return float(np.sum(returns.mean() * weights))


def weights_to_np(weights) :  # convert string of weights to numpy array
    return np.fromstring(weights, dtype=float, sep=' ')


def one_shot_optimization(data):
    result_df = pd.DataFrame()
    risk_targets = [0.03, 0.06, 0.09, 0.12, 0.15]

    for risk_target in risk_targets:
        list_per_class = []
        port_opt = portfolio_optimizer(min_weight=0, max_weight=1,
                                       cov_function="HC",
                                       freq="monthly",
                                       )
        port_opt.set_returns(data)
        weights = port_opt.optimize('meanVariance', risk_target)
        list_per_class.append(round(sum(weights[:4]), 3))# stocks
        list_per_class.append(round(sum(weights[4:6]),3))# commodities
        list_per_class.append(round(sum(weights[6:8]),3))# bonds
        list_per_class.append(round(weights[8],3))
        result_df[100 *risk_target] = list_per_class

    return result_df


def average_weights(data) :
    total_weight = np.zeros(9)
    for idx in range(len(data) - 1) :
        data_string = data.iloc[idx + 1][1 :-1]
        weight = weights_to_np(data_string)
        total_weight = np.add(total_weight, weight)
    weight_list = [x / (len(data) - 1) for x in total_weight.tolist()]
    return [round(num, 3) for num in weight_list]


def weights_df(files) :
    idx_level = [1, 4, 7, 10, 13]
    df = pd.DataFrame()
    counter = 0
    for idx in idx_level :
        for file in files :
            weights_per_class = []
            data = pd.read_csv(file, parse_dates=['date']). \
                       set_index(['date']).iloc[1 :]
            weights_subset = data.iloc[:, idx]
            av_weights = average_weights(weights_subset)
            weights_per_class.append(sum(av_weights[0 :4]))  # stocks
            weights_per_class.append(sum(av_weights[4 :6]))  # commodities
            weights_per_class.append(sum(av_weights[6 :8]))  # bonds
            weights_per_class.append(av_weights[8])  # real estate
            df[counter] = weights_per_class
            counter += 1
    return df



if __name__ == "__main__" :
    lookback_win = 10
    lookback_months = 120

    files_weights = ["C:\\Universität\\Numerical Methods\\without_%dyr_threshold0.50_GS1_weights.csv" % lookback_win,
                     "C:\\Universität\\Numerical Methods\\without_%dyr_threshold0.50_HC_weights.csv"% lookback_win,
                     "C:\\Universität\\Numerical Methods\\without_%dyr_threshold0.50_SM_weights.csv"% lookback_win,
                     "C:\\Universität\\Numerical Methods\\without_%dyr_threshold0.50_SM2_weights.csv" % lookback_win
                     ]

    ret = pd.read_csv("C:\\Universität\\Numerical Methods\\prcs.csv", parse_dates=['date']). \
        set_index(['date']).pct_change().dropna()

    #weights = pd.read_csv(files_weights[0], parse_dates=['date']). \
                  #set_index(['date']).iloc[1 :]  # drop first row (zero weights)
    one_shot_optimization(ret).to_csv("one_shot_weights.csv", index = False)
    #weights_df(files_weights).to_csv("restr_weights_%s.csv" % lookback_win)
