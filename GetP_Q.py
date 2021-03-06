#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


pool = pd.read_csv("pool.csv", encoding='utf-8')
pool = pool.set_index("Date")
pool = pool.fillna(pool.mean())
view = pd.read_csv("view.csv", encoding='utf-8')
view = view.set_index("Date")
view = view.fillna(view.mean())
view.head()


def getP_Q(t, cycle):
    # t: time point, must be larger than cycle
    # cycle: yearly:12, quaterly:4

    # WARNING, only two views for equity and bond
    equity_funds = (pool["Wells Fargo C&B Large Cap Value A"].iloc[t] - pool["Wells Fargo C&B Large Cap Value A"].iloc[
        t - cycle]) / pool["Wells Fargo C&B Large Cap Value A"].iloc[t - cycle]
    bond_funds = (pool["Metropolitan West Total Return Bd I"].iloc[t] -
                  pool["Metropolitan West Total Return Bd I"].iloc[t - cycle]) / \
                 pool["Metropolitan West Total Return Bd I"].iloc[t - cycle]
    equity_benchmark = (view["iShares Russell 1000 Value ETF"].iloc[t] - view["iShares Russell 1000 Value ETF"].iloc[
        t - cycle]) / view["iShares Russell 1000 Value ETF"].iloc[t - cycle]
    bond_benchmark = (view["iShares Core US Aggregate Bond ETF"].iloc[t] -
                      view["iShares Core US Aggregate Bond ETF"].iloc[t - cycle]) / \
                     view["iShares Core US Aggregate Bond ETF"].iloc[t - cycle]
    view_Q = equity_benchmark - bond_benchmark

    P = np.array([[1, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    Q = np.array([view_Q, 0, 0, 0, 0])

    # P.dot(expected_return) = Q + error
    return P, Q
