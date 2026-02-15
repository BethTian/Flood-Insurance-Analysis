import pandas as pd
import numpy as np

def read_file(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def dropna_row(df, cols=None):
    """
    Drop rows where any of the specified columns contain NaN.
    """
    if cols is None:
        return df.dropna()

    valid_cols = [c for c in cols if c in df.columns]

    if not valid_cols:
        return df

    return df.dropna(subset=valid_cols)

def state_filter(df, county_col_name, target_state="North Carolina"):
    "filter all states in North Carolina"
    df_nc = df[df[county_col_name].astype(str).str.strip().str.lower()
        == target_state.lower()]
    return df_nc

def dropna_with_threshold(df, cols=None, threshold=1.0, axis_indicator = 0):
    """
    threshold = 0.0  -> delete for any Nan
    threshold = 0.5  -> delete for over 50% Nan
    threshold = 1.0  -> delete for all Nan
    """
    if cols is None:
        check_cols = list(df.columns)
    else:
        check_cols = [c for c in cols if c in df.columns]
        if not check_cols:
            return df
    nan_ratio = df[check_cols].isna().mean(axis=0)
    drop_cols = nan_ratio[nan_ratio > threshold].index.tolist()
    return df.drop(columns = drop_cols)



def standard_county_column(df, county_col_name, standard_name = "COUNTY", suffix = None):
    df_COUNTY = df.rename(columns = {county_col_name: standard_name})
    if suffix:
        df_COUNTY['COUNTY'] = df_COUNTY['COUNTY'].str.replace(suffix, '', regex=False)
    df_COUNTY["COUNTY"] = df_COUNTY["COUNTY"].str.upper()
    return df_COUNTY   


def remove_unknown_county(df,unknown_name, county_col = "COUNTY"):
    df_removed = df[df[county_col] != unknown_name]
    return df_removed

def drop_columns(df, drop_list=None, prefixes=None, suffixes=None, keep_prefixes = None):
    if drop_list is None:
        drop_list = []
    # prefix
    if prefixes:
        if isinstance(prefixes, list):
            prefixes = tuple(prefixes)
        drop_list += [col for col in df.columns if col.startswith(prefixes)]
    # suffix
    if suffixes:
        if isinstance(suffixes, list):
            suffixes = tuple(suffixes)
        drop_list += [col for col in df.columns if col.endswith(suffixes)]
    # prevent duplication
    drop_list = list(set(drop_list))
    if keep_prefixes:
        drop_list = [c for c in drop_list if c not in ('E_HU', 'E_HH')]
    return df.drop(columns=drop_list, errors="ignore")

def calculate_and_add_cols(df,denom_col: str, numer_cols=None, numer_suffixes=None, new_suffix=None, keep_original_numer= True,keep_original_denom = True):
    out = df.copy()
    # choose numerator columns
    if numer_cols is None:
        if numer_suffixes is None:
            raise ValueError("Provide numer_cols or numer_suffixes.")
        suffixes = tuple(numer_suffixes) if isinstance(numer_suffixes, list) else (numer_suffixes,)
        numer_cols = [c for c in out.columns if isinstance(c, str) and c.endswith(suffixes)]
    # denominator (avoid divide-by-zero)
    denom = out[denom_col].replace(0, np.nan)
    # compute ratios, add new cols
    for c in numer_cols:
        new_c = f"{c}{new_suffix}"   
        out[new_c] = out[c] / denom
    #  optionally drop numerator cols (denom col never dropped)
    if not keep_original_numer:
        out.drop(columns=numer_cols, inplace=True, errors="ignore")
    if not keep_original_denom:
        out.drop(columns=[denom_col], inplace=True, errors="ignore")

    return out

def select_cols(df, selected_lst):
    df_selected = df[selected_lst]
    return df_selected