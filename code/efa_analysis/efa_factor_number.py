import pandas as pd
import numpy as np
import factor_analyzer as fa
import matplotlib.pyplot as plt

def get_eigenvalues(df):
    fa_test = fa.FactorAnalyzer(rotation=None)
    fa_test.fit(df)
    ev, _ = fa_test.get_eigenvalues()
    return ev

def get_n_factors_kaiser(df):
    ev = get_eigenvalues(df)
    n_factors = (ev > 1).sum()
    return int(n_factors), ev

def get_n_factor_scree(df, show=True, save_path=None):
    ev = get_eigenvalues(df)
    x = range(1, len(ev) + 1)

    plt.figure()
    plt.scatter(x, ev)
    plt.plot(x, ev)
    plt.axhline(y=1, linestyle='--')   # optional
    plt.xlabel("Number of Factors")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return ev

