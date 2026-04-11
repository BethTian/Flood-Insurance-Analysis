import pandas as pd
import numpy as np
import factor_analyzer as fa


def get_high_corr(df, threshold=0.9, top_n=30):
    corr = df.corr()
    
    high_corr = (
        corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    
    high_corr = high_corr[high_corr > threshold]
    
    if top_n:
        high_corr = high_corr.head(top_n)
    
    return high_corr

def get_kmo(df):    
    kmo_all, kmo_model = fa.calculate_kmo(df)

    kmo_per_variable = pd.Series(kmo_all, index=df.columns)
    return kmo_model, kmo_per_variable.sort_values()

def get_bartlett(df):
    chi_square, p_value = fa.calculate_bartlett_sphericity(df)
    return chi_square, p_value

def filter_data(df):
    deleted_lst = ["log1p_HU_PER_SQMI_z", "CRF_VALUE_z", "RISK_SPCTL_z",
               "log1p_BUILDVALUE_DENS_z", 
               "log1p_ALR_VALB_z", "RESL_SPCTL_z","EP_AFAM_z", "log1p_HH_PER_SQMI_z", "HRCN_ALRA_z", 
               "log1p_EP_GROUPQ_z", "HRCN_ALRB_z", "log1p_IFLD_RISKV_z", "log1p_EP_LIMENG_z",
               "log1p_POPU_DENS_z", "log1p_RISK_VALUE_z",
               "log1p_AGRIVALUE_DENS_z", "ALR_VALA_z"
               ]
    new_df = df.drop(columns = deleted_lst)
    return new_df

