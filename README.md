# POM-feature-drift
Here we have POCs and tests for bridging's automated data drift detection.

## data/ref_data_creation.ipynb


## scoring_script_test.ipynb (POC)
The purpose of this test is a proof of concept implementation of NannyML performance estimation into the bridging scoring script. This could then be used to analyse data drift.

We take the portion of the bridging scoring script that reads EOO data, cleans it and then scores it using the step-up TA and ARPU models. For each of these models, we bring in reference data created in 'data/ref_data_creation.ipynb' and run our NannyML performance estimation. This outputs a mean ROC_AUC score for the reference data, a mean ROC_AUC on the new EOO data, a percentage decrease, and an 'alert' that is True/False depending on the new ROC_AUC falling below a given threshold. 

This output is given as a Pandas DataFrame but in production could be exported as a BigQuery table that is amended each month when the script is ran. We could also use this 'alert' to notify those running the script and/or to trigger further analysis (such as data drift analysis).

## validation_data_test.ipynb (tests)
