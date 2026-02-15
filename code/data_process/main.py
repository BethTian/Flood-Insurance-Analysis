import pandas as pd
import clean
import numpy as np
import aggregate
import filter_policy as fp
import standardize

sys_path = "/Users/bethtian/Desktop/vscode_projects/Flood_insurance_analysis/data/raw_data/"
path_FIPS = sys_path+"FIPS_county_code_name/State,_County_and_City_FIPS_Reference_Table_20260123.csv"
path_policy = sys_path+"NC_NFIP_policies/FimaNfipPoliciesV2_NC.csv"
path_penetration = sys_path+"NFIP_penetration_rates/NfipResidentialPenetrationRates.csv"
path_NRI = sys_path+"NRI/NRI_Table_Counties.csv"
path_SVI = sys_path+"SVI/SVI_2022_US_county.csv"
ref_path = sys_path+"FIPS_county_code_name/State,_County_and_City_FIPS_Reference_Table_20260123.csv"
unrelated_cols_path = sys_path+"NC_NFIP_policies/unrelated_policy_cols.txt"

output_sys_path = "/Users/bethtian/Desktop/vscode_projects/Flood_insurance_analysis/data/data_after_process/"
output_path_policy = output_sys_path+"policy.csv"
output_path_penetration = output_sys_path+"penetration.csv"
output_path_NRI = output_sys_path+"NRI.csv"
output_path_SVI = output_sys_path+"SVI.csv"
output_path_dependent = output_sys_path+"dependent.csv"
output_path_independent = output_sys_path+"independent.csv"

def clean_penetration_data(path):
    df = clean.read_file(path)
    #filter the north carolina rows
    df = clean.state_filter(df,"state")
    df = clean.standard_county_column(df,"county")
    
    # select the goal columns
    selected_cols = ["COUNTY", "resPenetrationRate"]
    df = clean.select_cols(df,selected_cols)
    df = clean.remove_unknown_county(df, "UNKNOWN COUNTIES")
    return df


def clean_nri_data(path,):
    df = clean.read_file(path)
    df = clean.state_filter(df, county_col_name="STATE")
    
    df = clean.standard_county_column(df,county_col_name="County")
    
    #string is not useful for the analysis, so we delete all of them, but we keep COUNTY variable
    string_cols = df.select_dtypes(include="object").columns.drop("COUNTY")
    df = clean.drop_columns(df, drop_list = string_cols)

    #delete the data with specific suffixes
    suffixes = ('_NPCTL', '_SCORE', 'EALS', '_RISKS', '_EXPPE', '_AFREQ', '_EALPE', '_VALPE', 'EALT', 'EXPT', 'EAL_VALT',
                "_EXPB", "_EXPA", "_EXPP", "_HLRB", "_HLRP", "_HLRA", "_EALB", "_EALA", "_EALP", "_EVNTS")
    df = clean.drop_columns(df, suffixes = suffixes)

    #delete the columns with specidic prefixes
    prefixes = ["HAIL", "LTNG", "CWAV", "HWAV", "ISTM", "LNDS", "ERQK", "DRGT", "WFIR","WNTW", "SWND", "TRND",
                     "EAL_SPCTL", "EAL_VALB", "EAL_VALP", "EAL_VALA"]
    df = clean.drop_columns(df, prefixes=prefixes)

    # calculate the new columns and variables
    df = clean.calculate_and_add_cols(df, denom_col="AREA",numer_suffixes="_EXP_AREA",new_suffix="_RATIO",keep_original_numer=False)
    df = clean.calculate_and_add_cols(df, denom_col = "AREA", numer_cols=["POPULATION", "BUILDVALUE", "AGRIVALUE"],new_suffix="_DENS", keep_original_numer = False)

    #delete all nan columns
    df = clean.dropna_with_threshold(df, threshold= 0.5)

    #delete the columns that are duplicate with column county
    drop_cols_lst = ["OID_", "STATEFIPS",
                "COUNTYFIPS", "STCOFIPS"  ]
    df = clean.drop_columns(df, drop_list = drop_cols_lst)
    
    return df

def clean_svi_data(path,):
    df = clean.read_file(path)
    df = clean.state_filter(df, county_col_name="STATE")

    df = clean.standard_county_column(df,county_col_name="County",suffix = ' County')

    #delete data with specific suffixes
    prefixes =  ("M_","MP_", "F_",# margin and error columns
                 "EPL_", "E_", "RPL_", "SPL_" # remove the 1. estimation value 2.percentile value 
                 )
    keep_prefixes = ('E_HU', 'E_HH')
    df = clean.drop_columns(df, prefixes = prefixes, keep_prefixes=keep_prefixes)
    df = clean.calculate_and_add_cols(df, denom_col = "AREA_SQMI", numer_cols=['E_HU','E_HH'],new_suffix="_PER_SQMI", keep_original_numer = False,keep_original_denom = False)
    
    #delete the columns that are duplicate with column county
    drop_cols_lst = ["ST", "STATE","ST_ABBR", "STCNTY", "FIPS", "LOCATION" ]
    df = clean.drop_columns(df, drop_list = drop_cols_lst)
    return df

def clean_policy_data(path):
    df_policy = clean.read_file(path)
    
    # 1) filter the policy data
    df_policy = fp.county_code_to_name(ref_path=ref_path, df=df_policy)
    
    unrelated_cols = fp.load_column_list_from_txt(unrelated_cols_path)
    
    df_policy = clean.drop_columns(df_policy, drop_list=unrelated_cols)
        # delete the variables that are worship or state owned properties
    indicator_cols = ("houseOfWorshipIndicator", "stateOwnedIndicator", "agricultureStructureIndicator")
    df_policy = fp.filter_by_indicators(df_policy, indicators=indicator_cols, keep_original=False)
        # delete all building description codes that is clearly not the residential buildings
    df_policy = fp.filter_non_residential_by_building_desc(df_policy, keep_original = False)
        #  keep the residential occupancy properties
    df_policy = fp.keep_residential_by_occupancy(df_policy, keep_original = False)
    df_policy = fp.remove_invalid_cancellation_before_effective(df_policy)

    date_cols = ["policyEffectiveDate","policyTerminationDate","cancellationDateOfFloodPolicy","originalNBDate"]
    df_policy = fp.parse_date_columns(df_policy, date_cols)
    print(df_policy.columns)
    #aggregate the data into panel
    df = aggregate.build_county_year_panel(df_policy)
    df = aggregate.build_county_policy_final(df)

    Analysis_cols = ["COUNTY", "cancellation_rate","active_premium_per_policy",
    "new_premium_per_policy","canceled_premium_per_policy", "expired_premium_per_policy"]
    df = fp.select_cols(df,selected_lst=Analysis_cols)
    return df

def single_dataset_clean():
    # #NRI
    df_nri = clean_nri_data(path_NRI)
    df_nri.to_csv(output_path_NRI, index=False)
    print("Saved NRI")

    # penetration
    df_pen = clean_penetration_data(path_penetration)
    df_pen.to_csv(output_path_penetration, index=False)
    print("Saved Penetration")

    # svi
    df_svi = clean_svi_data(path_SVI)
    df_svi.to_csv(output_path_SVI, index=False)
    print("Saved SVI")

    #nri
    df_policy = clean_policy_data(path_policy)
    df_policy.to_csv(output_path_policy, index = False)
    print("Saved policy")

    return df_nri, df_pen, df_svi, df_policy


def dependent_dataset_process(df_policy, df_pen):
    # combine the policy data (dependent variables)
    merged_df= pd.merge(df_policy,df_pen, on =["COUNTY"])

    num_cols = merged_df.select_dtypes(include="number").columns

    no_log_cols = ["resPenetrationRate", "cancellation_rate"]
    log_cols = [c for c in num_cols if c not in no_log_cols]

    merged_df = standardize.log_transform(merged_df, log_cols, keep_original= False)
    num_cols = merged_df.select_dtypes(include="number").columns
    merged_df = standardize.zscore_transform(merged_df, num_cols, keep_original = False)
    merged_df.to_csv(output_path_dependent)
    return merged_df

def independent_dataset_process(df_nri, df_svi):
    # combine the policy data (dependent variables)
    merged_df= pd.merge(df_nri,df_svi, on =["COUNTY"])

    drop_cols = ["ALR_VALP","HRCN_ALRP","IFLD_ALRP", "EP_NHPI","IFLD_ALRA"]
    merged_df = clean.drop_columns(merged_df, drop_list=drop_cols)

    no_transform_cols = ["RISK_SPCTL","SOVI_SPCTL","RESL_SPCTL"]

    # use the z score to deal with columns
    zscore_cols = [
        "EP_POV150","EP_UNEMP","EP_HBURD","EP_NOHSDP",
        "EP_AGE65","EP_AGE17","EP_DISABL","EP_SNGPNT","EP_MINRTY",
        "EP_MOBILE","EP_CROWD","EP_NOVEH","EP_NOINT",
        "EP_AFAM","EP_HISP","EP_TWOMORE",
        "RESL_VALUE","CRF_VALUE",
        "ALR_VALA","HRCN_ALRB","HRCN_ALRA","IFLD_ALRB"
    ]
    merged_df = standardize.zscore_transform(merged_df, cols = zscore_cols, keep_original= False)

    candidate_transform_cols = [
        "RISK_VALUE","HRCN_RISKV","IFLD_RISKV",
        "POPULATION_DENS","BUILDVALUE_DENS","AGRIVALUE_DENS",
        "E_HU_PER_SQMI","E_HH_PER_SQMI",
        "EP_AIAN","EP_ASIAN","EP_NHPI","EP_GROUPQ","EP_MUNIT","EP_LIMENG",
        "IFLD_ALRA","IFLD_EXP_AREA_RATIO","EP_OTHERRACE",
        "EP_UNINSUR","ALR_VALB",
    ]
    merged_df = standardize.log_transform(merged_df, candidate_transform_cols, keep_original = False)

    merged_df["HRCN_UNEXP_RATIO"] = 1- pd.to_numeric(merged_df["HRCN_EXP_AREA_RATIO"])
    merged_df["log1p_HRCN_UNEXP_RATIO"] = np.log1p(merged_df["HRCN_UNEXP_RATIO"].fillna(0))
    merged_df.drop(["HRCN_UNEXP_RATIO","HRCN_EXP_AREA_RATIO" ], axis = 1, inplace=True)

    # use the zscore to deal with the log columns
    log_cols = [c for c in merged_df.columns if c.startswith("log1p_")]+no_transform_cols
    merged_df = standardize.zscore_transform(merged_df, cols = log_cols, keep_original= False)
    merged_df.to_csv(output_path_independent)
    return merged_df

if __name__ == "__main__":
    df_nri, df_pen, df_svi, df_policy = single_dataset_clean()
    df_dependent = dependent_dataset_process(df_policy, df_pen)
    df_independent = independent_dataset_process(df_nri, df_svi)