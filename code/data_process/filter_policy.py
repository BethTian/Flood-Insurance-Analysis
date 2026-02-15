import pandas as pd
import clean

def load_column_list_from_txt(path):
    with open(path, "r") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols

def select_cols(df, selected_lst):
    df_selected = df[selected_lst]
    return df_selected

def drop_filter_columns(df,cols) :
    """Drop columns used for filtering (ignore missing)."""
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def county_code_to_name(ref_path, df):

    df_ref = pd.read_csv(ref_path, usecols=["StCnty FIPS Code", "County Name"])
    df_ref = df_ref.rename(columns = {"StCnty FIPS Code": "countyCode", "County Name": "COUNTY"})
    df_merged = df.merge(df_ref, on = "countyCode", how = "left")
    df_merged_clean = clean.dropna_row( df_merged, cols = ['COUNTY',"femaRegion"])
    return df_merged_clean


def filter_by_indicators(df,indicators, drop_value=1, keep_original=True):
    """Remove rows where any indicator column equals drop_value (default 1)."""
    out = df.copy()
    for col in indicators:
        if col in out.columns:
            out = out[out[col] != drop_value]
    if not keep_original:
        out = out.drop(columns=[c for c in indicators if c in out.columns],
                       errors="ignore")
    return out


def filter_non_residential_by_building_desc(df, non_residential_codes=None,col="buildingDescriptionCode",keep_original= True):
    """Remove rows whose buildingDescriptionCode is in non_residential_codes."""
    if non_residential_codes is None:
        non_residential_codes = [
            3,  # Detached Garage
            4,  # Agricultural Building
            5,  # Warehouse
            6,  # Pool / Clubhouse / Recreation
            7,  # Tool / Storage Shed
            8,  # Other
            9,  # Barn
            14, # Commercial Building
            17, # House of Worship
            19  # Travel Trailer
        ]
    out = df.copy()
    if col in out.columns:
        out = out[~out[col].isin(non_residential_codes)]
        if not keep_original:
            out = out.drop(columns=[col], errors="ignore")

    return out


def keep_residential_by_occupancy(df, residential_codes=None,col="occupancyType",keep_original= True):
    """Keep rows whose occupancyType is in residential_codes."""
    if residential_codes is None:
        residential_codes = [1, 2, 11, 12, 14, 16]
    out = df.copy()
    if col in out.columns:
        out = out[out[col].isin(residential_codes)]
        if not keep_original:
            out = out.drop(columns=[col], errors="ignore")
    return out


def remove_invalid_cancellation_before_effective(df, cancellation_col="cancellationDateOfFloodPolicy",effective_col="policyEffectiveDate",):
    """Drop rows where cancellation date exists and is earlier than effective date."""
    out = df.copy()
    if cancellation_col in out.columns and effective_col in out.columns:
        mask_bad = out[cancellation_col].notna() & (out[cancellation_col] < out[effective_col])
        out = out[~mask_bad]
    return out


def parse_date_columns(df, date_cols):
    for col in date_cols:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)

            # If timezone-aware, remove timezone
            if isinstance(s.dtype, pd.DatetimeTZDtype):
                s = s.dt.tz_convert(None)

            df[col] = s

    return df