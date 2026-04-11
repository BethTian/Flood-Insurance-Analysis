import os
import pandas as pd
import factor_analyzer as fa

def get_factors_full(df, n, rotation_method='promax', loading_thre=0.6):
    fa_model = fa.FactorAnalyzer(n_factors=n, rotation=rotation_method)
    fa_model.fit(df)

    # 1. loadings
    loadings_df = pd.DataFrame(
        fa_model.loadings_,
        index=df.columns,
        columns=[f'Factor{i+1}' for i in range(n)]
    )

    # 2. variance explained
    variance_df = pd.DataFrame(
        fa_model.get_factor_variance(),
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=[f'Factor{i+1}' for i in range(n)]
    )

    # 3. factor scores
    factor_scores = pd.DataFrame(
        fa_model.transform(df),
        index=df.index,
        columns=[f'Factor{i+1}' for i in range(n)]
    )

    # 4. significant loadings
    sig_dict = {}
    for col in loadings_df.columns:
        sig = loadings_df.loc[loadings_df[col].abs() > loading_thre, col] \
            .sort_values(key=abs, ascending=False)
        sig_dict[col] = sig

    return loadings_df, variance_df, factor_scores, sig_dict


def save_results(df, n, rotation_method='promax', loading_thre=0.6, output_dir='.'):
    loadings_df, variance_df, factor_scores, sig_dict = get_factors_full(
        df=df,
        n=n,
        rotation_method=rotation_method,
        loading_thre=loading_thre
    )

    os.makedirs(output_dir, exist_ok=True)

    # save significant loadings
    non_empty_sig = {k: v for k, v in sig_dict.items() if not v.empty}
    if non_empty_sig:
        sig_df = pd.concat(non_empty_sig, names=['Factor', 'Variable']).reset_index()
        sig_df.columns = ['Factor', 'Variable', 'Loading']
        sig_df.to_csv(os.path.join(output_dir, "sig_variables.csv"), index=False)
    else:
        sig_df = pd.DataFrame(columns=['Factor', 'Variable', 'Loading'])
        sig_df.to_csv(os.path.join(output_dir, "sig_variables.csv"), index=False)

    # save factor scores
    factor_scores.to_csv(os.path.join(output_dir, "efa_factors.csv"))

    # save variance
    variance_df.to_csv(os.path.join(output_dir, "efa_variance.csv"))

    # optional: save full loadings too
    loadings_df.to_csv(os.path.join(output_dir, "efa_loadings.csv"))

    return loadings_df, variance_df, factor_scores, sig_df