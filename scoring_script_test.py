#**********************************************************
#**********************************************************
#
#                      $$$
#                      $$$
#                      $$$
#            $$$$$$$$  $$$   $$$  $$$         $$$
#           $$$$$      $$$  $$$    $$$       $$$
#          $$$$        $$$ $$$      $$$     $$$
#           $$$$$      $$$$$$$       $$$   $$$
#              $$$$$   $$$$$$$        $$$ $$$
#                $$$$  $$$  $$$        $$$$$
#              $$$$$   $$$   $$$        $$$
#          $$$$$$$$    $$$    $$$      $$$
#                                     $$$
#                                    $$$
#                                   $$$
#
#**********************************************************
#**********************************************************

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Project : POM selection process on GCP
#Step :    01-05 Import, tranform, export and load
#First release : April 2020
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

#The step can be divided in the below-mentioned tasks:
# 1. Delete all the files from the GCS folder
# 2. Create a temp table for offer_bridging_eoo_base in the working dataset
# 3. Move scoring dataset from BQ table to csv files
# 4. Load scoring dataset from GCS to pandas dataframe
#   *NB different preprocessing steps used for the new models, hence two different datasets*
#   *New models are the 18 DTV UK ones*
# 5. Score Arpu models
# 6. Score Churn models
# 7. Score TA models
# 8. Merge all the scores in one dataframe
# 9. Derive new columns - suppress_ro, opt_treat, rank
# 10. Set RO to 10% for DTV customers
# 11. Import base date and cohort and convert it into string to tag ouput tables from the workflow
# 12. Move scored data from pandas dataframe to BQ table
# 13. Create a table that can be used in the next steps by using scoring output

# pip install xgboost==1.0.2
# pip install lightgbm==3.3.2 DO THIS IN NEW MACHINE

import pickle
import numpy as np
import re
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt;
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pom_NEW import *
import os
import gcsfs
from google.cloud import storage
from google.cloud import bigquery
import project_config as pc
import common_variables as cv
import logging
# import pandas_gbq

logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("{0}/{1}.log".format('logs','logfile')),
                              logging.StreamHandler()]
                   )
logging.info('******** Execution of step 1-4 import data, transform data and score the models ************\n');

prefix_name = 'pom_scoring'

#Delete from the GCS folder
delete_blob_from_gcs(project_id = pc.project_id, bucket_name = pc.bucket, prefix_name = prefix_name)


table_id = 'offer_bridging_eoo_base'
bucket_location = 'EU'
bucket_id       = 'gs://'+pc.bucket+'/pom_scoring'
file_name       = 'eoo_base_' + str(dt.datetime.now().date())
file_format     = 'CSV'
gcs_file_path   = os.path.join(bucket_id,file_name+'_*.csv')


#Connection to BQ
client = bigquery.Client(project=pc.project_id) #;

#Create a temp table for offer_bridging_eoo_base
#Below query is to fix difference in offer end calendar weeks
query = """
Declare inp_cohort string;

Set inp_cohort =
(select Cast(min(cd.Comcast_Week_And_Year) as string) || '-' ||  Cast(max(cd.Comcast_Week_And_Year) as string)
from uk_inp_olivemirror_ic.offer_bridging_eoo_base eoo
    inner join
    uk_inp_olivemirror_ic.WH_Calendar_Dim cd
    on cd.calendar_date = eoo.Earliest_Offer_End_Dt
where acc_offer_rnk = 1);

create or replace table """+pc.target_dataset+""".offer_bridging_eoo_base as
select eoo.* Except(Comcast_Earliest_Offer_End_Wk, Cohort, p_true_touch_type)
        , Cast(p_true_touch_type as string) as p_true_touch_type
        , cd.Comcast_Week_And_Year
        , cd.Comcast_Week_And_Year as Comcast_Earliest_Offer_End_Wk
        , inp_cohort as Cohort
        
        , cvsd.cancels_1mth, cvsd.cancels_3mth, cvsd.cancels_6mth, cvsd.cancels_9mth, cvsd.cancels_12mth
        , cvsd.mobile_1mth, cvsd.mobile_3mth, cvsd.mobile_6mth, cvsd.mobile_9mth, cvsd.mobile_12mth
        , cvsd.bill_1mth, cvsd.bill_3mth, cvsd.bill_6mth, cvsd.bill_9mth, cvsd.bill_12mth
        , cvsd.signin_1mth, cvsd.signin_3mth, cvsd.signin_6mth, cvsd.signin_9mth, cvsd.signin_12mth
        , cvsd.vip_1mth, cvsd.vip_3mth, cvsd.vip_6mth, cvsd.vip_9mth, cvsd.vip_12mth
        , cvsd.help_1mth, cvsd.help_3mth, cvsd.help_6mth, cvsd.help_9mth, cvsd.help_12mth
        , cvsd.contact_1mth
        , cvsd.shop_12mth
        , cvsd.issues_6mth
        , cvsd.issues_sat_6mth
        , cvsd.issues_wifi_6mth
        , cvsd.complains_6mth, cvsd.complains_9mth, cvsd.complains_12mth
        , cvsd.web_winback_3mth
        
        , cvsd.total_viewing_duration_overlap_3m, cvsd.total_viewing_duration_overlap_2m, cvsd.total_viewing_duration_overlap_1m
        , cvsd.linear_viewing_total_overlap_1m, cvsd.linear_viewing_total_overlap_2m, cvsd.linear_viewing_total_overlap_3m
        , cvsd.total_sky_cinema_duration_overlap_3m, cvsd.total_sky_cinema_duration_overlap_2m, cvsd.total_sky_cinema_duration_overlap_1m
        , cvsd.total_sky_sports_duration_overlap_3m, cvsd.total_sky_sports_duration_overlap_2m, cvsd.total_sky_sports_duration_overlap_1m
from 
uk_inp_olivemirror_ic.offer_bridging_eoo_base eoo 
    inner join
    uk_inp_olivemirror_ic.WH_Calendar_Dim cd
    on cd.calendar_date = eoo.Earliest_Offer_End_Dt
LEFT JOIN  """+pc.target_dataset+""".eoo_click_and_view AS cvsd
            ON cvsd.account_number = eoo.account_number
where acc_offer_rnk = 1

"""
query_job = client.query(query) #;
query_job.result() #;
logging.info('Create a temp table for offer_bridging_eoo_base') #;

#Move scoring data from BQ to GCS
bq_to_gcs(project = pc.project_id, bq_dataset = pc.target_dataset, bq_table = table_id,gcs_file_path = gcs_file_path, gcs_location = bucket_location, gcs_file_format = file_format)
logging.info('Move scoring data from BQ to GCS') #;

#Connection to BQ
client = bigquery.Client(project=pc.project_id) #;
query = """ SELECT column_name, data_type
FROM """+pc.target_dataset+""".INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'offer_bridging_eoo_base'
AND data_type = 'DATE'
"""

date_cols = client.query(query).to_dataframe().iloc[:, 0].tolist()

#Load columns used in new models - speeds up process as we don't need to keep unused columns
cols = {'Account_Number', 'base_dt', 'eoo_base_obs_dt', 'Cohort', 'Rack_Rate', 'Ttl_Offer_Discount', 'Customer_Type', 'Country'}
for customer_type in ['DTV', 'SABB']:
    for country in ['UK']:
        for target_type in ['arpu', 'churn', 'ta', 'bbcoe']:
            for model_type in ['NT', 'L', 'M', 'H', 'R', 'NX', 'SU', 'RO']:
                pickle_name = f'pickle_files/{customer_type}_{country}_{target_type}_{model_type}.pkl'
                if os.path.isfile(pickle_name):
                    with open(pickle_name, 'rb') as pickle_file:
                        model = pickle.load(pickle_file)
                    if (type(model) is XGBRegressor) or (type(model) is XGBClassifier):
                        model_columns = model.get_booster().feature_names
                    elif (type(model) is LGBMRegressor) or (type(model) is LGBMClassifier):
                        model_columns = model.feature_name_
                    elif (type(model) is CombinedModel):
                        model_columns = model.feature_name_
                    else:
                        print(f'MODEL TYPE NOT MATCHED {pickle_name}')
                        model_columns = []
                    if model_columns is None:
                        print(f'NONE COLUMNS {pickle_name}')
                        model_columns = []
                    cols = cols.union(model_columns)
rename_andrew = fix_columns()
cols = cols.union({x for x, y in rename_andrew.items() if y in cols})
cols = cols.union({x.rsplit('_', 1)[0] for x in cols})

#Load scoring data from GCS to pandas dataframe
df_main = read_pomdata_to_score(project = pc.project_id, bucket_name = pc.bucket, prefix_name = prefix_name)
logging.info('Load scoring data from GCS to pandas dataframe') #;

logging.info('*****************************Initializing the class****************************************')
dpp = DataPreProcess(df=df_main.rename(columns={'EOO_Base_Obs_Dt' : 'eoo_base_obs_dt'}), cols=cols, date_cols=date_cols)

del df_main

logging.info('*****************************Processing the dates******************************************')
_ = dpp.process_dates()

logging.info('*****************************Imputing missing values***************************************')
_, _, _, _ = dpp.fill_missing()

logging.info('*****************************Create Scaled Numerics******************************************')
_ = dpp.scale_numeric(excl_cols=['Account_Number'])

logging.info('*****************************One Hot Encode values (NEW)***************************************')
_ = dpp.one_hot_encode(nunique=100)
                     
logging.info('*****************************Consolidating data***************************************')

data_consolidated = dpp.concat_data(column_fix={'HD_Product_Holding_nan' : 'HD_Product_Holding_None'})
data_consolidated_scaled = dpp.concat_data(scale_numeric=True, column_fix=rename_andrew)

data_consolidated_uk_dtv = data_consolidated.loc[(data_consolidated['Country_UK'] == 1) & (data_consolidated['Customer_Type'] == 'DTV')]
data_consolidated_uk_dtv_scaled = data_consolidated_scaled.loc[(data_consolidated_scaled['Country_UK'] == 1) & (data_consolidated_scaled['Customer_Type'] == 'DTV')]
data_consolidated_uk_sabb = data_consolidated.loc[(data_consolidated['Country_UK'] == 1) & (data_consolidated['Customer_Type'] == 'SABB')]

data_consolidated_roi_dtv = data_consolidated.loc[(data_consolidated['Country_UK'] == 0) & (data_consolidated['Customer_Type'] == 'DTV')]
data_consolidated_roi_dtv_scaled = data_consolidated_scaled.loc[(data_consolidated_scaled['Country_UK'] == 0) & (data_consolidated_scaled['Customer_Type'] == 'DTV')]

stepup_models = ['NT', 'L', 'M', 'H']
rollover_models = ['NX', 'R']
sabb_models = ['NT', 'RO', 'SU']

logging.info('*****************************Begin scoring***************************************')

#Score arpu models
data_dict_uk_dtv_stepup = score_data(df=data_consolidated_uk_dtv_scaled, customer_type='DTV', country='UK', target='target_arpu', model_types=stepup_models)
data_dict_uk_dtv_rollover = score_data(df=data_consolidated_uk_dtv, customer_type='DTV', country='UK', target='target_arpu', model_types=rollover_models)
data_dict_uk_sabb = score_data(df=data_consolidated_uk_sabb, customer_type='SABB', country='UK', target='target_arpu', model_types=sabb_models)

data_dict_roi_dtv_stepup = score_data(df=data_consolidated_roi_dtv_scaled, customer_type='DTV', country='ROI', target='target_arpu', model_types=stepup_models)
data_dict_roi_dtv_rollover = score_data(df=data_consolidated_roi_dtv, customer_type='DTV', country='ROI', target='target_arpu', model_types=rollover_models)

#Check arrays of account numbers are equal to each other across treatment types

assert((data_dict_uk_dtv_stepup['NT'].index.values == data_dict_uk_dtv_stepup['L'].index.values).all and       (data_dict_uk_dtv_stepup['L'].index.values == data_dict_uk_dtv_stepup['M'].index.values).all and       (data_dict_uk_dtv_stepup['M'].index.values == data_dict_uk_dtv_stepup['H'].index.values).all and       (data_dict_uk_dtv_stepup['H'].index.values == data_dict_uk_dtv_rollover['R'].index.values).all and       (data_dict_uk_dtv_rollover['R'].index.values == data_dict_uk_dtv_rollover['NX'].index.values).all
      )
assert((data_dict_roi_dtv_stepup['NT'].index.values == data_dict_roi_dtv_stepup['L'].index.values).all and       (data_dict_roi_dtv_stepup['L'].index.values == data_dict_roi_dtv_stepup['M'].index.values).all and       (data_dict_roi_dtv_stepup['M'].index.values == data_dict_roi_dtv_stepup['H'].index.values).all and       (data_dict_roi_dtv_stepup['H'].index.values == data_dict_roi_dtv_rollover['R'].index.values).all and       (data_dict_roi_dtv_rollover['R'].index.values == data_dict_roi_dtv_rollover['NX'].index.values).all
      )
assert((data_dict_uk_sabb['NT'].index.values == data_dict_uk_sabb['RO'].index.values).all and       (data_dict_uk_sabb['RO'].index.values == data_dict_uk_sabb['SU'].index.values).all
      )

scored_df_arpu_uk_dtv_new = pd.concat(list(data_dict_uk_dtv_stepup.values()), axis=1)
scored_df_arpu_uk_dtv_existing = pd.concat(list(data_dict_uk_dtv_rollover.values()), axis=1)
scored_df_arpu_uk_dtv = pd.concat([scored_df_arpu_uk_dtv_new, scored_df_arpu_uk_dtv_existing], axis=1)

scored_df_arpu_roi_dtv_new = pd.concat(list(data_dict_roi_dtv_stepup.values()), axis=1)
scored_df_arpu_roi_dtv_existing = pd.concat(list(data_dict_roi_dtv_rollover.values()), axis=1)
scored_df_arpu_roi_dtv = pd.concat([scored_df_arpu_roi_dtv_new, scored_df_arpu_roi_dtv_existing], axis=1)

scored_df_arpu_uk_sabb = pd.concat(list(data_dict_uk_sabb.values()), axis=1)

scored_df_arpu = pd.concat([scored_df_arpu_uk_dtv, scored_df_arpu_roi_dtv, scored_df_arpu_uk_sabb])

del scored_df_arpu_uk_dtv_new
del scored_df_arpu_uk_dtv_existing
del scored_df_arpu_uk_dtv
del scored_df_arpu_roi_dtv_new
del scored_df_arpu_roi_dtv_existing
del scored_df_arpu_roi_dtv
del scored_df_arpu_uk_sabb
del data_dict_uk_dtv_stepup
del data_dict_uk_dtv_rollover
del data_dict_roi_dtv_stepup
del data_dict_roi_dtv_rollover
del data_dict_uk_sabb

logging.info('Score arpu models');

#Churn Scoring
data_dict_uk_dtv = score_data(df=data_consolidated_uk_dtv, customer_type='DTV', country='UK', target='target_churn', model_types=stepup_models+rollover_models)
data_dict_uk_sabb = score_data(df=data_consolidated_uk_sabb,customer_type='SABB', country='UK',  target='target_churn', model_types=sabb_models)

data_dict_roi_dtv = score_data(df=data_consolidated_roi_dtv, customer_type='DTV', country='ROI', target='target_churn', model_types=stepup_models+rollover_models)

#Check arrays of account numbers are equal to each other across treatment types
assert((data_dict_uk_dtv['NT'].index.values == data_dict_uk_dtv['L'].index.values).all and       (data_dict_uk_dtv['L'].index.values == data_dict_uk_dtv['M'].index.values).all and       (data_dict_uk_dtv['M'].index.values == data_dict_uk_dtv['H'].index.values).all and       (data_dict_uk_dtv['H'].index.values == data_dict_uk_dtv['R'].index.values).all and       (data_dict_uk_dtv['R'].index.values == data_dict_uk_dtv['NX'].index.values).all
      )
assert((data_dict_roi_dtv['NT'].index.values == data_dict_roi_dtv['L'].index.values).all and       (data_dict_roi_dtv['L'].index.values == data_dict_roi_dtv['M'].index.values).all and       (data_dict_roi_dtv['M'].index.values == data_dict_roi_dtv['H'].index.values).all and       (data_dict_roi_dtv['H'].index.values == data_dict_roi_dtv['R'].index.values).all and       (data_dict_roi_dtv['R'].index.values == data_dict_roi_dtv['NX'].index.values).all
      )
assert((data_dict_uk_sabb['NT'].index.values == data_dict_uk_sabb['RO'].index.values).all and       (data_dict_uk_sabb['RO'].index.values == data_dict_uk_sabb['SU'].index.values).all
      )

scored_df_churn_uk_dtv = pd.concat(list(data_dict_uk_dtv.values()), axis=1)
scored_df_churn_roi_dtv = pd.concat(list(data_dict_roi_dtv.values()), axis=1)
scored_df_churn_uk_sabb = pd.concat(list(data_dict_uk_sabb.values()), axis=1)

scored_df_churn = pd.concat([scored_df_churn_uk_dtv, scored_df_churn_roi_dtv, scored_df_churn_uk_sabb])

del scored_df_churn_uk_dtv
del scored_df_churn_roi_dtv
del scored_df_churn_uk_sabb
del data_dict_uk_dtv
del data_dict_roi_dtv
del data_dict_uk_sabb

logging.info('Score churn models');

#Score TA models
data_dict_uk_dtv_stepup = score_data(df=data_consolidated_uk_dtv_scaled, customer_type='DTV', country='UK', target='target_ta', model_types=stepup_models)
data_dict_uk_dtv_rollover = score_data(df=data_consolidated_uk_dtv, customer_type='DTV', country='UK', target='target_ta', model_types=rollover_models)
data_dict_uk_sabb = score_data(df=data_consolidated_uk_sabb, customer_type='SABB', country='UK', target='target_bbcoe', model_types=sabb_models)

data_dict_roi_dtv_stepup = score_data(df=data_consolidated_roi_dtv_scaled, customer_type='DTV', country='ROI', target='target_ta', model_types=stepup_models)
data_dict_roi_dtv_rollover = score_data(df=data_consolidated_roi_dtv, customer_type='DTV', country='ROI', target='target_ta', model_types=rollover_models)

#Check arrays of account numbers are equal to each other across treatment types
assert((data_dict_uk_dtv_stepup['NT'].index.values == data_dict_uk_dtv_stepup['L'].index.values).all and       (data_dict_uk_dtv_stepup['L'].index.values == data_dict_uk_dtv_stepup['M'].index.values).all and       (data_dict_uk_dtv_stepup['M'].index.values == data_dict_uk_dtv_stepup['H'].index.values).all and       (data_dict_uk_dtv_stepup['H'].index.values == data_dict_uk_dtv_rollover['R'].index.values).all and       (data_dict_uk_dtv_rollover['R'].index.values == data_dict_uk_dtv_rollover['NX'].index.values).all
      )
assert((data_dict_roi_dtv_stepup['NT'].index.values == data_dict_roi_dtv_stepup['L'].index.values).all and       (data_dict_roi_dtv_stepup['L'].index.values == data_dict_roi_dtv_stepup['M'].index.values).all and       (data_dict_roi_dtv_stepup['M'].index.values == data_dict_roi_dtv_stepup['H'].index.values).all and       (data_dict_roi_dtv_stepup['H'].index.values == data_dict_roi_dtv_rollover['R'].index.values).all and       (data_dict_roi_dtv_rollover['R'].index.values == data_dict_roi_dtv_rollover['NX'].index.values).all
      )
assert((data_dict_uk_sabb['NT'].index.values == data_dict_uk_sabb['RO'].index.values).all and       (data_dict_uk_sabb['RO'].index.values == data_dict_uk_sabb['SU'].index.values).all
      )

scored_df_ta_uk_dtv_new = pd.concat(list(data_dict_uk_dtv_stepup.values()), axis=1)
scored_df_ta_uk_dtv_existing = pd.concat(list(data_dict_uk_dtv_rollover.values()), axis=1)
scored_df_ta_uk_dtv = pd.concat([scored_df_ta_uk_dtv_new, scored_df_ta_uk_dtv_existing], axis=1)

scored_df_ta_roi_dtv_new = pd.concat(list(data_dict_roi_dtv_stepup.values()), axis=1)
scored_df_ta_roi_dtv_existing = pd.concat(list(data_dict_roi_dtv_rollover.values()), axis=1)
scored_df_ta_roi_dtv = pd.concat([scored_df_ta_roi_dtv_new, scored_df_ta_roi_dtv_existing], axis=1)

scored_df_bbcoe = pd.concat(list(data_dict_uk_sabb.values()), axis=1)

scored_df_ta = pd.concat([scored_df_ta_uk_dtv, scored_df_ta_roi_dtv])

del scored_df_ta_uk_dtv_new
del scored_df_ta_uk_dtv_existing
del scored_df_ta_uk_dtv
del scored_df_ta_roi_dtv_new
del scored_df_ta_roi_dtv_existing
del scored_df_ta_roi_dtv
del data_dict_uk_dtv_stepup
del data_dict_uk_dtv_rollover
del data_dict_roi_dtv_stepup
del data_dict_roi_dtv_rollover
del data_dict_uk_sabb

# scored_df_ta.head()
logging.info('Score TA models');

scoring_df_vars_uk_dtv = data_consolidated_uk_dtv[['Account_Number', 'Rack_Rate', 'Ttl_Offer_Discount', 'Country_UK', 'Customer_Type']].set_index('Account_Number')
scoring_df_vars_uk_sabb = data_consolidated_uk_sabb[['Account_Number', 'Rack_Rate', 'Ttl_Offer_Discount', 'Country_UK', 'Customer_Type']].set_index('Account_Number')
scoring_df_vars_roi_dtv = data_consolidated_roi_dtv[['Account_Number', 'Rack_Rate', 'Ttl_Offer_Discount', 'Country_UK', 'Customer_Type']].set_index('Account_Number')

scoring_df_vars = pd.concat([scoring_df_vars_uk_dtv, scoring_df_vars_roi_dtv, scoring_df_vars_uk_sabb])

# scoring_df_vars.head()

del data_consolidated_uk_dtv_scaled
del data_consolidated_uk_dtv
del data_consolidated_uk_sabb
del data_consolidated_roi_dtv_scaled
del data_consolidated_roi_dtv

assert((scoring_df_vars.index.values == scored_df_arpu.index.values).all \
and (scoring_df_vars.index.values == scored_df_churn.index.values).all \
and (scoring_df_vars.loc[(scoring_df_vars['Customer_Type'] == 'DTV')].index.values == scored_df_ta.index.values).all \
and (scoring_df_vars.loc[(scoring_df_vars['Customer_Type'] == 'SABB')].index.values == scored_df_bbcoe.index.values).all)

#Merge all the scores in one dataframe
df_list = [scored_df_arpu, scored_df_churn, scored_df_ta, scoring_df_vars, scored_df_bbcoe]
df_all = pd.concat(df_list, axis=1).reset_index()
df_all.head()
logging.info('Merge all the scores in one dataframe');

# df_all.shape

with open('xchng_rate_gbp2eur.txt', 'r') as file:
    xchng_rate_gbp2eur = file.read().replace('\n', '')

xchng_rate_gbp2eur = float(xchng_rate_gbp2eur)

#Net Revenue change at EOO+22 Weeks

#ROI requires currency rate exchagen for pred_ta and pred_churn variables (not for arpu as this is already converted to Sterling)

df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_R'] = df_all['pred_arpu_R'] - 20 * df_all['pred_ta_R'] - 150 * df_all['pred_churn_R']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_NT'] = df_all['pred_arpu_NT'] - 20 * df_all['pred_ta_NT'] - 150 * df_all['pred_churn_NT']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_L'] = df_all['pred_arpu_L'] - 20 * df_all['pred_ta_L'] - 150 * df_all['pred_churn_L']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_M'] = df_all['pred_arpu_M'] - 20 * df_all['pred_ta_M'] - 150 * df_all['pred_churn_M']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_H'] = df_all['pred_arpu_H'] - 20 * df_all['pred_ta_H'] - 150 * df_all['pred_churn_H']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_NX'] = df_all['pred_arpu_NX'] - 20 * df_all['pred_ta_NX'] - 150 * df_all['pred_churn_NX']
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_NT'] = df_all['pred_arpu_NT'] - (20 * df_all['pred_ta_NT'] + 150 * df_all['pred_churn_NT']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_L'] = df_all['pred_arpu_L'] - (20 * df_all['pred_ta_L'] + 150 * df_all['pred_churn_L']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_M'] = df_all['pred_arpu_M'] - (20 * df_all['pred_ta_M'] + 150 * df_all['pred_churn_M']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_H'] = df_all['pred_arpu_H'] - (20 * df_all['pred_ta_H'] + 150 * df_all['pred_churn_H']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_R'] = df_all['pred_arpu_R'] - (20 * df_all['pred_ta_R'] + 150 * df_all['pred_churn_R']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 0) & (df_all['Customer_Type'] == 'DTV'), 'pred_rev_NX'] = df_all['pred_arpu_NX'] - (20 * df_all['pred_ta_NX'] + 150 * df_all['pred_churn_NX']) * xchng_rate_gbp2eur
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'SABB'), 'pred_rev_NT'] = df_all['pred_arpu_NT'] - 10 * df_all['pred_bbcoe_NT'] - 50 * df_all['pred_churn_NT']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'SABB'), 'pred_rev_SU'] = df_all['pred_arpu_SU'] - 10 * df_all['pred_bbcoe_SU'] - 50 * df_all['pred_churn_SU']
df_all.loc[(df_all['Country_UK'] == 1) & (df_all['Customer_Type'] == 'SABB'), 'pred_rev_RO'] = df_all['pred_arpu_RO'] - 10 * df_all['pred_bbcoe_RO'] - 50 * df_all['pred_churn_RO']
df_all['pred_rev_change_R'] = df_all['pred_rev_R'] - df_all['pred_rev_NX']
df_all['pred_rev_change_L'] = df_all['pred_rev_L'] - df_all['pred_rev_NT']
df_all['pred_rev_change_M'] = df_all['pred_rev_M'] - df_all['pred_rev_NT']
df_all['pred_rev_change_H'] = df_all['pred_rev_H'] - df_all['pred_rev_NT']
df_all['pred_rev_change_RO'] = df_all['pred_rev_RO'] - df_all['pred_rev_NT']
df_all['pred_rev_change_SU'] = df_all['pred_rev_SU'] - df_all['pred_rev_NT']

df_all['suppress_ro'] = 0
df_all.loc[df_all['Ttl_Offer_Discount']/df_all['Rack_Rate'] > 0.5, 'suppress_ro'] = 1
df_all.loc[df_all['suppress_ro']==1,'pred_rev_change_R'] = np.nan

treatment_dict = {'pred_rev_change_R' : ('RO', 10)
                  , 'pred_rev_change_H' : ('H', 9)
                  , 'pred_rev_change_M' : ('M', 7)
                  , 'pred_rev_change_L' : ('L', 5)
                  , 'pred_rev_change_RO' : ('RO', 5)
                  , 'pred_rev_change_SU' : ('SU', 5)}

df_all['pred_rev_change_opt'] = df_all[treatment_dict.keys()].max(axis=1)
df_all[['opt_treat', 'opt_cost']] = pd.DataFrame(data=list(df_all[treatment_dict.keys()].idxmax(axis=1).apply(lambda x: treatment_dict[x]))
                                                 , columns=['opt_treat', 'opt_cost'], index=df_all.index)

no_ro_treatment_dict = {'pred_rev_change_H' : ('H', 9)
                        , 'pred_rev_change_M' : ('M', 7)
                        , 'pred_rev_change_L' : ('L', 5)
                        , 'pred_rev_change_SU' : ('SU', 5)}

df_all['_2nd_pred_rev_change'] = df_all[no_ro_treatment_dict.keys()].max(axis=1)
df_all[['_2nd_treat', '_2nd_cost']] = pd.DataFrame(data=list(df_all[no_ro_treatment_dict.keys()].idxmax(axis=1).apply(lambda x: no_ro_treatment_dict[x]))
                                                 , columns=['_2nd_treat', '_2nd_cost'], index=df_all.index)

df_all['Rank'] = df_all['pred_rev_change_opt'].rank(ascending=False,method='first')

df_all[['final_treat', 'cost', 'pred_rev_change']] = df_all[['opt_treat', 'opt_cost', 'pred_rev_change_opt']]

#################################################
## Temp fix to increase/decrease RO to ~10% for DTV customers

df_all['pred_rev_diff_R'] = df_all['pred_rev_change_R'] - df_all['pred_rev_change_opt']
df_all['pred_rev_diff_SU'] = df_all['_2nd_pred_rev_change'] - df_all['pred_rev_change_opt']

for i, country in enumerate(['ROI', 'UK']):
    df_all[f'Rank_DTV_{country}'] = np.nan
    type_mask = (df_all.Customer_Type=='DTV') & (df_all.Country_UK==i)
    df_all.loc[type_mask, f'Rank_DTV_{country}'] = df_all.loc[type_mask, 'pred_rev_change_opt'].rank(ascending=False, pct=True, method='first')
    
    mask = type_mask & (df_all[f'Rank_DTV_{country}']<=float(cv.DTV_Selection_Threshold))
    ro_target = int(df_all[mask].shape[0] * 0.1)
    ro_current = (df_all.loc[mask, 'final_treat']=='RO').sum()
    
    if ro_current < ro_target:
        logging.info(f'Artificially increase RO in the top {int(100*float(cv.DTV_Selection_Threshold))}% for {country} from {ro_current} to {ro_target}');
        nlarg = df_all[mask].nlargest(ro_target, 'pred_rev_diff_R', keep='all').index
        df_all.loc[nlarg, ['final_treat', 'cost', 'pred_rev_change']] = 'RO', 10, df_all.loc[nlarg, 'pred_rev_change_R']
        
    elif ro_current > ro_target:
        logging.info(f'Artificially decrease RO in the top {int(100*float(cv.DTV_Selection_Threshold))}% for {country} from {ro_current} to {ro_target}');
        nlarg = df_all[mask & (df_all.opt_treat=='RO')].nlargest(ro_current - ro_target, 'pred_rev_diff_SU', keep='all').index
        df_all.loc[nlarg, ['final_treat', 'cost', 'pred_rev_change']] = df_all.loc[nlarg, ['_2nd_treat', '_2nd_cost', '_2nd_pred_rev_change']].values

##################################################

#Import base date and cohort and convert it into string to tag ouput tables from the workflow
base_date,cohort = cv.get_base_dt_and_cohort() #;

base_date = str(base_date) #;
base_date = base_date.replace('-','') #;
cohort = str(cohort) #;
cohort = cohort[4:6]+'_'+cohort[11:13] #;

#Move scored data from pandas dataframe to BQ table
#df_all.to_gbq(pc.target_dataset+'.POM_scoring_output_'+cohort+'_'+base_date, project_id=pc.project_id, if_exists='replace')
df_all.to_gbq(pc.target_dataset+'.POM_scoring_output', project_id=pc.project_id, if_exists='replace')

logging.info('Move scored data from pandas dataframe to BQ table\n') #;

date_suffix = dt.datetime.utcfromtimestamp(int(time.time())).strftime('%Y%m%d')

#Create a table that can be used in the next steps by using scoring output
# query = """create or replace table """+pc.target_dataset+""".POM_"""+base_date+"""_"""+cohort+"""_Scoring_Output_""" + date_suffix + """ as
# query = """create or replace table """+pc.target_dataset+""".POM_"""+base_date+"""_"""+cohort+"""_Scoring_Output as
query = """create or replace table """+pc.target_dataset+""".POM_"""+base_date+"""_"""+cohort+"""_Scoring_Output as
select
cast(Account_Number as string) as account_number
, Customer_Type as customer_type
, case when Country_UK=1 then 'UK' else 'ROI' end as Country

, pred_arpu_nx
, pred_arpu_r as pred_arpu_ex
, pred_arpu_nt as pred_arpu_ctrl
, pred_arpu_h as pred_arpu_high
, pred_arpu_m as pred_arpu_mid
, pred_arpu_l as pred_arpu_low

, pred_contribution_ntro --as pred_contribution_ex
, pred_contribution_ro --as pred_contribution_ex
, pred_contribution_ntsu --as pred_contribution_ctrl
, pred_contribution_su

, pred_churn_nx
, pred_churn_r
, pred_churn_nt 
, pred_churn_h
, pred_churn_m 
, pred_churn_l

, pred_churn_ntro
, pred_churn_ro
, pred_churn_ntsu
, pred_churn_su

, pred_ta_nx
, pred_ta_r as pred_ta_ex
, pred_ta_nt as pred_ta_ctrl
, pred_ta_h as pred_ta_high
, pred_ta_m as pred_ta_mid
, pred_ta_l as pred_ta_low

, pred_bbcoe_ntro
, pred_bbcoe_ro
, pred_bbcoe_ntsu
, pred_bbcoe_su

, pred_rev_NX
, pred_rev_R
, pred_rev_NT
, pred_rev_H
, pred_rev_M
, pred_rev_L

, pred_rev_NTSU
, pred_rev_SU
, pred_rev_NTRO
, pred_rev_RO

, pred_rev_change_R
, pred_rev_change_H
, pred_rev_change_M
, pred_rev_change_L
, pred_rev_change_RO
, pred_rev_change_SU

, pred_rev_change_opt
, opt_treat

, pred_rev_change
, final_treat as unified_treatment
, Rank as unified_account_rank

from """+pc.target_dataset+""".POM_scoring_output""" #;

query_job = client.query(query)
query_job.result()

#Create backup table
query = """create or replace table """+pc.target_dataset+""".POM_"""+base_date+"""_"""+cohort+"""_Scoring_Output_""" + date_suffix + """ as
select * from """+pc.target_dataset+""".POM_"""+base_date+"""_"""+cohort+"""_Scoring_Output"""

query_job = client.query(query)
query_job.result()

logging.info('POM_scoring_output_cohort_base_date has been created\n') #;
logging.info('Execution of first 5 steps has ended\n') #;