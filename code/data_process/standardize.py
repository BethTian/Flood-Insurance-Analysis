import numpy as np
from sklearn.preprocessing import StandardScaler


def log_transform(df, cols, keep_original=True, suffix="_log"):
    out = df.copy()
    valid_cols = [c for c in cols if c in out.columns]
    if not valid_cols:
        return out

    for col in valid_cols:
        out[col + suffix] = np.log1p(out[col])

    if not keep_original:
        out.drop(columns=valid_cols, inplace=True)
    return out


def zscore_transform(df, cols, keep_original=True, suffix="_z"):
    out = df.copy()
    valid_cols = [c for c in cols if c in out.columns]

    if not valid_cols:
        return out
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(out[valid_cols])

    scaled_cols = [c + suffix for c in valid_cols]
    out[scaled_cols] = scaled_array

    if not keep_original:
        out.drop(columns=valid_cols, inplace=True)

    return out

