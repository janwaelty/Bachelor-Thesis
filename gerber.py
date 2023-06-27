"""
Name    : gerber.py
Author  : Yinsen Miao
Contact : yinsenm@gmail.com
Time    : 7/1/2021
Desc    : Compute Gerber Statistics
"""
import numpy as np
import pandas as pd
from scipy import stats

def is_psd_def(cov_mat):
    """
    :param cov_mat: covariance matrix of p x p
    :return: true if positive semi definite (PSD)
    """
    return np.all(np.linalg.eigvals(cov_mat) > -1e-6)


def gerber_cov_stat0(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 0, orginal Gerber statistics, not always PSD
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 > threshold > 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                    
            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (pos + neg)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cov_mat, cor_mat



def gerber_cov_stat1(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 1
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    assert 1 >= threshold >= 0, "threshold shall between 0 and 1"
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    cov_mat = np.zeros((p, p))  # store covariance matrix
    cor_mat = np.zeros((p, p))  # store correlation matrix

    for i in range(p):
        for j in range(i + 1):
            neg = 0
            pos = 0
            nn = 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                        ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1

            # compute Gerber correlation matrix
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
            cov_mat[i, j] = cor_mat[i, j] * sd_vec[i] * sd_vec[j]
            cov_mat[j, i] = cov_mat[i, j]
    return cov_mat, cor_mat


def gerber_cov_stat2(rets: np.array, threshold: float=0.5) -> tuple:
    """
    compute Gerber covariance Statistics 2
    :param rets: assets return matrix of dimension n x p
    :param threshold: threshold is between 0 and 1
    :return: Gerber covariance matrix of p x p
    """
    n, p = rets.shape
    sd_vec = rets.std(axis=0)
    U = np.copy(rets)
    D = np.copy(rets)

    # update U and D matrix
    for i in range(p):
        U[:, i] = U[:, i] >= sd_vec[i] * threshold
        D[:, i] = D[:, i] <= -sd_vec[i] * threshold

    # update concordant matrix
    N_CONC = U.transpose() @ U + D.transpose() @ D

    # update discordant matrix
    N_DISC = U.transpose() @ D + D.transpose() @ U
    H = N_CONC - N_DISC
    h = np.sqrt(H.diagonal())

    # reshape vector h and sd_vec into matrix
    h = h.reshape((p, 1))
    sd_vec = sd_vec.reshape((p, 1))

    cor_mat = H / (h @ h.transpose())
    cov_mat = cor_mat * (sd_vec @ sd_vec.transpose())
    return cov_mat, cor_mat


# test gerber_cov_stat1 and gerber_cov_stat2
if __name__ == "__main__":
    bgn_date = "1988-01-30"
    end_date = "2020-12-31"
    nassets = 9
    file_path = "C:\\Universität\\Numerical Methods\\prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['date'], index_col=["date"]).pct_change()[bgn_date: end_date].iloc[:, 0: nassets]
    print(gerber_cov_stat2(rets_df.to_numpy()))




