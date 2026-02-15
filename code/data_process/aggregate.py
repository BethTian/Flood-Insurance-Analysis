import pandas as pd
import numpy as np


def add_policy_end_date(df,
                        cancel_col="cancellationDateOfFloodPolicy",
                        term_col="policyTerminationDate",
                        out_col="policy_end_date"):
    """
    policy_end_date = cancellationDate if exists else terminationDate
    """
    out = df.copy()
    out[out_col] = out[cancel_col].fillna(out[term_col])
    return out


def build_county_year_panel(df,
                            county_col="COUNTY",
                            premium_col="totalInsurancePremiumOfThePolicy",
                            eff_col="policyEffectiveDate",
                            cancel_col="cancellationDateOfFloodPolicy",
                            term_col="policyTerminationDate",
                            end_col="policy_end_date",
                            years=range(2020, 2026),
                            fill_missing_with_zero=True,
                            create_end_date = True):
    """
    Build COUNTY x YEAR panel with:
      active/new/canceled/expired counts + premium sums,
      active_lag and cancellation_rate
    """
    if create_end_date and (end_col not in df.columns):
        df[end_col] = df[cancel_col].fillna(df[term_col])

    records = []

    def agg(mask, prefix):
        return (
            df.loc[mask]
            .groupby(county_col)
            .agg(
                **{
                    f"{prefix}_count": (premium_col, "size"),
                    f"{prefix}_premium": (premium_col, "sum"),
                }
            )
            .reset_index()
        )

    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y}-12-31")

        # Active during year: effective <= year_end and end_date >= year_start
        mask_active = (df[eff_col] <= end) & (df[end_col] >= start)

        # Newly effective in year
        mask_new = df[eff_col].dt.year == y
        # Canceled in year
        mask_canceled = df[cancel_col].notna() & (df[cancel_col].dt.year == y)

        # Expired in year (termination in year AND not canceled before termination)
        term_mask = df[term_col].notna() & (df[term_col].dt.year == y)
        cancelled_before_term = df[cancel_col].notna() & (df[cancel_col] < df[term_col])
        mask_expired = term_mask & (~cancelled_before_term)

        merged = (
            agg(mask_active, "active")
            .merge(agg(mask_new, "new"), on=county_col, how="outer")
            .merge(agg(mask_canceled, "canceled"), on=county_col, how="outer")
            .merge(agg(mask_expired, "expired"), on=county_col, how="outer")
        )

        if fill_missing_with_zero:
            merged = merged.fillna(0)

        # types
        count_cols = [c for c in merged.columns if c.endswith("_count")]
        prem_cols = [c for c in merged.columns if c.endswith("_premium")]
        merged[count_cols] = merged[count_cols].astype(int)
        merged[prem_cols] = merged[prem_cols].astype(float)

        merged["year"] = y
        records.append(merged)

    panel = (
        pd.concat(records, ignore_index=True)
        .sort_values([county_col, "year"])
        .reset_index(drop=True)
    )

    panel["active_lag"] = panel.groupby(county_col)["active_count"].shift(1)
    panel["cancellation_rate"] = panel["canceled_count"] / panel["active_lag"]
    panel["cancellation_rate"] = panel["cancellation_rate"].replace([np.inf, -np.inf], np.nan)

    return panel


def build_county_policy_final(panel,
                             county_col="COUNTY",
                             base_year=2022,
                             rate_years=(2021, 2025),
                             pairs=("active", "new", "canceled", "expired")):
    """
    Final county-level output:
      - snapshot counts & premiums in base_year
      - avg cancellation_rate across rate_years
      - premium_per_policy for each pair
    """
    start_y, end_y = rate_years

    # avg cancellation rate (e.g., 2021-2025)
    df_rate = panel[panel["year"].between(start_y, end_y)]
    cancel_rate = (
        df_rate.groupby(county_col)["cancellation_rate"]
        .mean()
        .reset_index()
    )

    # base year snapshot
    keep_cols = [county_col]
    for p in pairs:
        keep_cols += [f"{p}_count", f"{p}_premium"]

    base = panel[panel["year"] == base_year][keep_cols]
    final = base.merge(cancel_rate, on=county_col, how="left")

    # premium per policy
    for p in pairs:
        final[f"{p}_premium_per_policy"] = (
            final[f"{p}_premium"] / final[f"{p}_count"].replace(0, np.nan)
        ).fillna(0)

    final[[f"{p}_premium_per_policy" for p in pairs]] = (
        final[[f"{p}_premium_per_policy" for p in pairs]]
        .replace([np.inf, -np.inf], 0)
    )

    return final
