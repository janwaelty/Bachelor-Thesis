# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:21:58 2021
@author: Patrick Ledoit
"""


# function sigmahat=cov1Para(Y,k)
#
# Y (N*p): raw data matrix of N iid observations on p random variables
# sigmahat (p*p): invertible covariance matrix estimator
#
# Shrinks towards one-parameter matrix:
#    all variances of the target are the same
#    all covariances of the target are zero
#
# If the second (optional) parameter k is absent, not-a-number, or empty,
# then the algorithm demeans the data by default, and adjusts the effective
# sample size accordingly. If the user inputs k = 0, then no demeaning
# takes place; if (s)he inputs k = 1, then it signifies that the data x has
# already been demeaned.
#
# This version: 01/2021, based on the 04/2014 version

###########################################################################
# This file is released under the BSD 2-clause license.

# Copyright (c) 2014-2021, Olivier Ledoit and Michael Wolf
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###########################################################################
def cov1Para(Y, k=None) :
    # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    # Post-Condition: Sigmahat dataframe is returned

    import numpy as np
    import pandas as pd
    import math
    # de-mean returns if required
    Y= Y.copy()
    # Y = pd.DataFrame(d).astype(float).pct_change().dropna()
    Y = pd.DataFrame(Y).astype(float)
    # de-mean returns if required
    N, p = Y.shape  # sample size and matrix dimension

    # default setting
    if k is None or math.isnan(k) :
        mean = Y.mean(axis=0)
        Y = Y.sub(mean, axis=1)  # demean
        k = 1

    # vars
    n = N - k  # adjust effective sample size

    # Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n

    # compute shrinkage target
    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag) / len(diag)
    target = meanvar * np.eye(p)

    # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n  # sample covariance matrix of squared returns
    piMat = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))

    pihat = sum(piMat.sum())

    # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
    gammahat = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2

    # diagonal part of the parameter that we call rho
    rho_diag = 0;

    # off-diagonal part of the parameter that we call rho
    rho_off = 0;

    # compute shrinkage intensity
    rhohat = rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0, min(1, kappahat / n))

    # compute shrinkage estimator
    sigmahat = shrinkage * target + (1 - shrinkage) * sample

    return sigmahat.to_numpy(), shrinkage

if __name__ == "__main__":
    import pandas as pd
    bgn_date = "1990-01-29"
    end_date = "2020-01-01"
    nassets = 9
    file_path = "C:\\Universität\\Numerical Methods\\prcs.csv"
    rets_df = pd.read_csv(file_path, parse_dates=['date'], index_col=["date"]).pct_change()[bgn_date: end_date].iloc[:, 0: nassets]
    rets = rets_df.values
    covMat,shrinkage = cov1Para(rets)
    print(covMat)




