# POM-feature-drift
Here we have POCs and tests for bridging's automated data drift detection.

## data/ref_data_creation.ipynb
This notebook creates the reference data that is used in both scoring_script_test.ipynb and validation_data_test.ipynb. To create the reference data we read in the relevant csvs from a GCS bucket, create a timestamp column using the week from the index, drop unnecessary scoring columns, and order by timestamps. This is done for the full five validation cohorts and for the first three validation cohohorts for the different uses (detailed in sections below).

We also pre-fit and pickle the data drift calculators using model training data as reference. This allows us to save runtime in below scripts by only having to load the calculator and use it on the new data.

## scoring_script_test.ipynb (POC)
The purpose of this notebook is a proof of concept implementation of NannyML performance estimation into the bridging scoring script. This could then be used to analyse data drift.

We take the portion of the bridging scoring script that reads EOO data, cleans it and then scores it using the step-up TA and ARPU models. For each of these models, we bring in reference data created in 'data/ref_data_creation.ipynb' and run our NannyML performance estimation. This outputs a mean ROC_AUC score for the reference data, a mean ROC_AUC on the new EOO data, a percentage decrease, and an 'alert' that is True/False depending on the new ROC_AUC falling below a given threshold. 

This output is given as a Pandas DataFrame but in production could be exported as a BigQuery table that is amended each month when the script is ran. We could also use this 'alert' to notify those running the script and/or to trigger further analysis (such as data drift analysis).

## validation_data_test.ipynb (tests)
This notebook is for experimenting with NannyML's uses. It is split into tests for NT and Low treatment TA and ARPU. We first setup data for analysis. For the purpose of this test we use the final two validation cohorts that were used during the validation of the current TA and ARPU models. We then bring in the reference data from data/ref_data_creation.ipynb. For this test we use the first three validation cohorts as our reference.

We then proceed to run the NannyML performance estimation and show how we can use the plots built into the package. Then we compare the estimation to the ground truth. The end of each section contains data drift plots and calculations that utilise the calculators created in data/ref_data_creation.ipynb and plot the drift of each feature in order of LGBM feature importance.