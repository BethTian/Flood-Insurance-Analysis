import pandas as pd
import efa_factor_number
import efa_model
import efa_test
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

file_path = "/Users/bethtian/Desktop/vscode_projects/Flood_insurance_analysis/data/data_after_process/independent.csv"
save_path = "/Users/bethtian/Desktop/vscode_projects/Flood_insurance_analysis/data/data_after_efa"
def read_file(f_path):
    df = pd.read_csv(f_path, index_col=0)
    df = df.drop(columns = ["AREA"])
    df = df.set_index("COUNTY")
    return df

def get_test(df):
    high_corr = efa_test.get_high_corr(df, threshold=0.5)
    print(f"The correlations between variables are: \n{high_corr}")
    
    kmo_model, kmo_all = efa_test.get_kmo(df)
    print("KMO (overall):", kmo_model)
    print(f"KMO (variables): \n{kmo_all}")

    chi_square, p_value = efa_test.get_bartlett(df)
    print(f"The barlett test chi square is {chi_square}, and P value is {p_value}.")
    return high_corr, kmo_model, kmo_all, chi_square, p_value

def filter_data(df, lst):
    new_df = df.drop(columns = lst)
    return new_df

def get_n_factor(df, show_scree=True, save_path=None):
    n_kaiser, ev = efa_factor_number.get_n_factors_kaiser(df)

    if show_scree:
        efa_factor_number.get_n_factor_scree(df, show=True, save_path=save_path)

    return {
        "kaiser_n_factors": n_kaiser,
        "eigenvalues": ev
    }

def run_efa(df, n, rotation_method='promax', loading_thre=0.6,
            save=False, output_dir='.'):
    
    if save:
        loadings_df, variance_df, factor_scores, sig_dict = efa_model.save_results(
        df = df, n = n, rotation_method = "promax", output_dir = output_dir
    )
    else:
        loadings_df, variance_df, factor_scores, sig_dict = efa_model.get_factors_full(
            df = df, n = n, rotation_method = "promax"
        )
    
    return loadings_df, variance_df, factor_scores, sig_dict

if __name__ == "__main__":
    # read file
    df = read_file(file_path)

    #filter data based on the correlation and tests for efa
    # high_corr, kmo_model, kmo_all, chi_square, p_value = get_test(df)
    deleted_lst = ["E_HU_PER_SQMI_log", "CRF_VALUE_z", "RISK_SPCTL_z",
            "BUILDVALUE_DENS_log", 
            "ALR_VALB_log", "RESL_SPCTL_z","EP_AFAM_z", "E_HH_PER_SQMI_log", "HRCN_ALRA_z", 
            "EP_GROUPQ_log", "HRCN_ALRB_z", "IFLD_RISKV_log", "EP_LIMENG_log",
            "POPULATION_DENS_log", "RISK_VALUE_log",
            "AGRIVALUE_DENS_log", "ALR_VALA_z"
            ]
    new_df = filter_data(df, deleted_lst)
    high_corr, kmo_model, kmo_all, chi_square, p_value = get_test(new_df)

    # #select the factor number 
    # dct = get_n_factor(new_df)
    # print(f"The number of factor from Kaiser criterion is {dct['kaiser_n_factors']}")
    
    run_efa(new_df, 5, rotation_method='promax', loading_thre=0.6,
            save=True, output_dir=save_path)
    
    factor_name = ['Housing ','socioeconomic vulnerability',
                   'Age structure', 'Hazard Risk Exposure',
                    'Insurance Vulnerability']