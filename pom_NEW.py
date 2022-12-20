import numpy as np
import pandas as pd
import re
import pickle
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import matplotlib.pyplot as plt;
import os
import gcsfs
from google.cloud import storage
from google.cloud import bigquery
import multiprocessing
num_cpu = multiprocessing.cpu_count()

class DataPreProcess(object):
    def __init__(self, df, cols=[], date_cols=[]):
        self.df = df.loc[:, df.columns.intersection(cols)]
        self.imputed_values = {}
        self.label_encoding = {}
        
        if (len(date_cols) > 0):
            date_cols = self.df.columns.intersection(date_cols)
            self.df.loc[:, date_cols] = df.loc[:, date_cols].apply(pd.to_datetime, errors='coerce')

        if 'base_dt' in self.df.columns.to_list():
            print("base_dt available in the date df")
            self.base_date = 'base_dt'
        elif 'eoo_base_obs_dt' in self.df.columns.to_list():
            print("eoo_base_obs_dt available in the date df")
            self.base_date = 'eoo_base_obs_dt'
        else:
            assert('base_dt'=='eoo_base_obs_dt')
       
        self.num_df = self.df.select_dtypes(include = ['int8', 'int32', 'int64', 'float'])
        self.cat_df = self.df.select_dtypes(include = 'object')
        self.date_df = self.df.select_dtypes(include = 'datetime')
        assert(len(self.df.columns) == len(self.num_df.columns) + len(self.cat_df.columns) + len(self.date_df.columns))

    def process_dates(self):

        for col in self.date_df.drop(self.base_date, axis = 1).columns:
                self.date_df[col + '_monthdiff'] =  ((self.date_df[self.base_date] - self.date_df[col])/np.timedelta64(1, 'M')).apply(np.floor)
                self.date_df.drop(col, axis = 1, inplace = True)
        self.date_df.drop(self.base_date, axis = 1, inplace = True)

        return self.date_df
    
    def scale_numeric(self, excl_cols=[]): # FIX THIS
        """
        Scale numeric columns with sklearn standard scaler
        @param model_data : Pandas dataframe for cleaning.
        @param excl_cols: Columns to be excluded from the scaling
        @return: cleaned dataframe
        """
        self.num_df_scaled = self.num_df.copy()
        
        num_cols = self.num_df_scaled.columns.to_list()
        num_cols = [col for col in num_cols if col not in excl_cols]

        scaler = StandardScaler()

        self.num_df_scaled[num_cols] = scaler.fit_transform(self.num_df_scaled[num_cols])

        return self.num_df_scaled

    def fill_missing(self, missing_num=None, missing_obj=None):

#         #TODO: Apply exception in the code if imputed values are found in the data

#         op_list = []
#         for i in self.num_df.columns:
#             if self.num_df.loc[self.num_df[i] == missing_num, i].count() > 0:
#                 op_list.append(i)         
        
#         for i in self.date_df.columns:
#             if self.date_df.loc[self.date_df[i] == missing_num, i].count() > 0:
#                 op_list.append(i)
        
#         for i in self.cat_df.columns:     
#             if self.cat_df.loc[self.cat_df[i] == missing_obj, i].count() > 0:
#                 op_list.append(i)

#         print("variable list containing imputation values: {}".format(op_list))
#         if len(op_list) == 0:
#             print("Filling missing values")
#             self.num_df = self.num_df.fillna(missing_num)
#             self.date_df = self.date_df.fillna(missing_num)
#             self.cat_df = self.cat_df.fillna(missing_obj)

#         for col in self.num_df.columns:
#             self.imputed_values[col] = missing_num

#         for col in self.date_df.columns:
#             self.imputed_values[col] = missing_num

#         for col in self.cat_df.columns:
#             self.imputed_values[col] = missing_obj

        return self.num_df, self.date_df, self.cat_df, self.imputed_values

    def one_hot_encode(self, drop=None, nunique=0, excl_cols=['Cohort', 'Account_Number', 'Customer_Type', 'Treatment_Type']):
        """
        One hot encode categorical columns 

        @param model_data : Pandas dataframe for cleaning
        @return: cleaned dataframe and list of encoded columns
        """
        cat_cols_w_nunique_gt_assigned = self.cat_df.columns[self.cat_df.apply(pd.Series.nunique) > nunique].to_list() if nunique > 0 else []
        print(cat_cols_w_nunique_gt_assigned)
        
        for col in excl_cols:
            if col in cat_cols_w_nunique_gt_assigned:
                cat_cols_w_nunique_gt_assigned.remove(col)

        if len(cat_cols_w_nunique_gt_assigned) > 0:
            print("Columns with unique values > {} are {}".format(nunique, cat_cols_w_nunique_gt_assigned))
            self.cat_df.drop(columns=cat_cols_w_nunique_gt_assigned, inplace=True)
        
        if not hasattr(self, 'df_cohort'):
            for col in excl_cols:
                if col in self.cat_df.columns:
                    self.df_cohort = self.cat_df[col]
                    self.cat_df.drop(columns=col, inplace=True)

        ohe = OneHotEncoder(drop=drop)

        df_dummies = pd.DataFrame(ohe.fit_transform(self.cat_df).toarray(), columns=ohe.get_feature_names_out(self.cat_df.columns), index=self.cat_df.index) 
        df_dummies = df_dummies.rename(columns=replace_special_chars)
        self.cat_enc_df = pd.concat([self.df_cohort, df_dummies], axis=1)
        
        return self.cat_enc_df

    
    def concat_data(self, scale_numeric=False, column_fix={}):
    
        num_df = self.num_df_scaled if scale_numeric else self.num_df    
        
        _df = pd.concat([num_df, self.date_df, self.cat_enc_df], axis = 1)

        print("Data contains the total {} rows and {} columns".format(*_df.shape))
        
        for old_col, new_col in column_fix.items():
            if old_col in _df.columns:
                _df[new_col] = _df[old_col]
            else:
                print(old_col, new_col)
        return _df



# ############################################################################################################

class CombinedModel:
    def __init__(self, model_list):
        self.model_list_ = model_list
        self.feature_name_ = list(set().union(*[model.feature_name_ for model in model_list]))
        
        coefs_dict = {feature : [] for feature in self.feature_name_}
        for model in model_list:
            for feature, coef in zip(model.feature_name_, model.feature_importances_):
                coefs_dict[feature].append(coef)
        self.feature_importances_ = [sum(c)/len(c) for c in coefs_dict.values()]
    
    def predict_proba(self, data):
        pred_sum = sum(model.predict_proba(data[model.feature_name_]) for model in self.model_list_)
        #print(f'{pred_sum')
        return(pred_sum)
    
# ############################################################################################################

def delete_blob_from_gcs(project_id, bucket_name, prefix_name):
    storage_client = storage.Client(project = project_id)
    bucket = storage_client.get_bucket(bucket_name)
    files = bucket.list_blobs(prefix = prefix_name)
    fileList = [file.name for file in files if '.' in file.name]

    print("total blobs that would be deleted are {}".format(fileList))

    if len(fileList) > 0:
        for file in fileList:
            bucket.delete_blob(file)

def bq_to_gcs(project, bq_dataset, bq_table, \
              gcs_file_path, gcs_location, gcs_file_format):
    # setting up bq
    bq = bigquery.Client(project=project)
    dataset_ref = bq.dataset(bq_dataset)
    table_ref = dataset_ref.table(bq_table)

    # setting up gcs
    job_config = bigquery.job.ExtractJobConfig()
    job_config.destination_format = gcs_file_format
    extract_job = bq.extract_table(table_ref,
                                   gcs_file_path,
                                   location=gcs_location,
                                   job_config=job_config)
    extract_job.result()

def read_pomdata_to_score(project, bucket_name, prefix_name):

    fs = gcsfs.GCSFileSystem(project=project)
    files=fs.ls (os.path.join(bucket_name, prefix_name))
    files = [file for file in files if file.endswith('.csv')]
    print("blobs are {}".format(files))

    if len(files) > 0:
        df_init = pd.read_csv("gs://" + files[0])

    dtypes = pd.DataFrame(df_init.dtypes).reset_index().rename({'index': 'column', 0: 'dtype'}, axis = 1)
    date_variables = list(dtypes.loc[(dtypes['dtype'] == 'object') & (dtypes['column'].str.contains('_d', flags=re.IGNORECASE, \
                                                                                                regex=True)),'column'])
    print("Total # of date variables are {}".format(len(date_variables)))

    df_main = pd.concat(map(lambda x: pd.read_csv("gs://" + x, encoding = "utf-8", parse_dates = date_variables,\
                                                  infer_datetime_format = True, dayfirst = True, low_memory = False), files))

    print("Inital dataframe contains {} rows and {} columns".format(*df_main.shape))

    # rename each column by removing special chars
    df_main.rename(columns = lambda x: x.replace('.', ''), inplace = True)

    print("Shape of the dataset is {0} rows and {1} columns".format(*df_main.shape))
    return df_main

def score_data(df,customer_type, country, target, model_types=['NT', 'L', 'M', 'H', 'R', 'NX','SU']):
    target_type = target.split('_', 1)[1]
    data_dict = {}
    for model_type in model_types:
        model_name = f'{customer_type}_{country}_{target_type}_{model_type}'
        pickle_name = f'pickle_files/{model_name}.pkl'
        if os.path.isfile(pickle_name):
            print (f"************************Model file {model_name} exists************************")

            with open(pickle_name, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
    
            if (type(model) is XGBRegressor) or (type(model) is XGBClassifier):
                model_columns = model.get_booster().feature_names
            elif (type(model) is LGBMRegressor) or (type(model) is LGBMClassifier):
                model_columns = model.feature_name_
            elif (type(model) is CombinedModel):
                model_columns = model.feature_name_
            else:
                print('MODEL TYPE NOT MATCHED')
                model_columns = []

            dat = df.loc[:, df.columns.intersection(model_columns)]
    
            missing_dummies = list(set(model_columns).difference(set(dat.columns)))
            if missing_dummies:
                print("Missing dummies: {}".format(missing_dummies))
    
            for col in missing_dummies:
                # print(f'{col} missing, 0 imputed')
                dat[col] = 0
    
            dat = dat[model_columns] # same ordering required as the existing model
    
            if target_type in ('arpu','contribution'):
                y_pred = model.predict(dat)
            elif target_type in ('churn', 'ta', 'bbcoe'):
                y_pred = model.predict_proba(dat)[:, 1]
            else:
                print(f'Target variable {target} can not be scored')
    
            var_name = f'pred_{target_type}_{model_type}'
    
            data_dict[model_type] = pd.DataFrame(data = {'Account_Number': df.Account_Number.values, var_name: y_pred}).set_index('Account_Number')
            
            print(f'************************Scored data for model {model_name}************************')
        
        else:
            print (f'************************Model file {pickle_name} DOES NOT exist************************')
    return data_dict

def replace_special_chars(string, replace_with='', chars={'<' : 'LE', '.' : '_'}):
    """
    Remove special characters from a string 

    @param string: string which we are cleaning
    @replace_with: default replacement value 
    @chars: dictionary of special cases 
    @return: cleaned string
    """

    for key,value in chars.items():
        string = string.replace(key, value)

        string = re.sub(r'[\\\\ \"\'\][/<>,&)(\-=+.%$£!^*~@{};:?`|]', replace_with, string)

    return string

def fix_columns():
    fix_new_models = {
        'Age': 'age',
        'ARPU': 'arpu',
        'BB_Churns_In_Last_3Yr' : 'bb_churns_in_last_3yr',
        'BB_Product_Holding_FibreMax' : 'bb_product_holding_FibreMax',
        'BB_Product_Holding_UnlimitedFibre' : 'bb_product_holding_UnlimitedFibre',
        'BB_Subscriber_Activations_In_Last_3Yr' : 'bb_subber_activns_in_last_3yr',
        'Curr_Contract_Offer_Amount_BB' : 'curr_contract_offer_amount_bb',
        'Curr_Offer_Amount_DTV': 'curr_offer_amount_dtv',
        'Curr_Offer_Amount_HD': 'curr_offer_amount_hd',
        'Curr_Offer_Amount_LR': 'curr_offer_amount_lr',
        'Curr_Offer_Amount_Movies' : 'curr_offer_amount_movies',
        'Curr_Offer_Amount_SKY_BOX_SETS' : 'curr_offer_amount_sky_box_sets',
        'Curr_Offer_Amount_SKY_KIDS' : 'curr_offer_amount_sky_kids',
        'Curr_Offer_Bridged_DTV_Rollover' : 'curr_offer_bridged_dtv_Rollover',
        'Curr_Offer_Start_Dt_DTV_monthdiff' : 'curr_offer_start_dt_dtv_monthdiff',
        'Curr_Offer_Subscription_Sub_Type_Sports_SPORTS' : 'curr_offr_sub_type_sports_SPORTS',
        'DTV_Activations_In_Last_3Yr': 'dtv_activations_in_last_3yr',
        'DTV_Last_Activation_Dt_monthdiff': 'dtv_last_activation_dt_monthdiff',
        'DTV_Last_Active_Block_Dt_monthdiff' : 'dtv_last_active_block_dt_monthdiff',
        'DTV_Last_PC_Effective_To_Dt_monthdiff': 'dtv_last_pc_effective_to_dt_monthdiff',
        'HD_Active': 'hd_active',
        'HD_Product_Holding_HDBasicSkyHD': 'hd_product_holding_HDBasicSkyHD',
        'HD_Product_Holding_HDPremiumRose': 'hd_product_holding_HDPremiumRose',
        'Last_All_Call_Dt_monthdiff' : 'last_all_call_dt_monthdiff',
        'Last_Completed_OD_DL_Dt_monthdiff': 'last_completed_od_dl_dt_monthdiff',
        'Last_Credit_Dt_monthdiff' : 'last_credit_dt_monthdiff',
        'Last_Offer_Applied_Dt_DTV_monthdiff': 'last_offer_applied_dt_dtv_monthdiff',
        'Last_Service_Call_Dt_monthdiff' : 'last_service_call_dt_monthdiff',
        'last_TA_dt_monthdiff': 'last_ta_dt_monthdiff',
        'last_TA_outcome_TurnaroundSaved' : 'last_ta_outcome_TurnaroundSaved',
        'last_TA_reason_FinancialSituation' : 'last_ta_reason_FinancialSituation',
        'Last_Value_Call_Dt_monthdiff': 'last_value_call_dt_monthdiff',
        'Lima_ICD_Flag' : 'lima_icd_flag',
        'max_speed_uplift': 'max_speed_uplift',
        'Movies_Active' : 'movies_active',
        'MS_Active' : 'ms_active',
        'OD_DLs_Completed_In_Last_7d': 'od_dls_completed_in_last_7d',
        'Offers_Applied_Lst_24M_DTV' : 'offers_applied_lst_24m_dtv',
        'Prev_Offer_Amount_BB' : 'prev_offer_amount_bb',
        'Prev_Offer_Amount_DTV': 'prev_offer_amount_dtv',
        'Prev_Offer_Amount_LR' : 'prev_offer_amount_lr',
        'Prev_Offer_Amount_Movies' : 'prev_offer_amount_movies',
        'Prev_Offer_Subscription_Sub_Type_Sports_SPORTS' : 'prev_offr_sub_type_sports_SPORTS', 
        'SGE_Product_Holding_SGEPaid': 'sge_product_holding_SGEPaid',
        'Sky_Consumer_Market_Share' : 'sky_consumer_market_share',
        'Sports_Active': 'sports_active',
        'Sports_Product_Count' : 'sports_product_count',
        'Talk_Product_Holding_SkyPayAsYouTalk': 'talk_product_holding_SkyPayAsYouTalk',
        'TAs_in_last_24m' : 'tas_in_last_24m',
        'Throughput_Speed': 'throughput_speed',
        'Ttl_Offer_Discount': 'ttl_offer_discount',
        'Virgin_Consumer_Market_Share': 'virgin_consumer_market_share'
    }

    return fix_new_models

########################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
