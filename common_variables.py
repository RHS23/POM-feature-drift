from google.cloud import bigquery
import project_config as pc

#level1 = '0.40';

DTV_Selection_Threshold = '0.40';
SABB_Selection_Threshold = '0.50';

def get_base_dt_and_cohort():
	base_dt = "";
	cohort = ""
	client = bigquery.Client(project=pc.project_id);
	query = """ select EOO_Base_Obs_dt as base_dt,cohort from """+pc.target_dataset+""".POM_DTV_UK_OOT_202120-202123 group by EOO_Base_Obs_dt,cohort """ 
	query_job = client.query(query);
	a = query_job.result();
	for i in a:
		base_dt=i.base_dt
		cohort = i.cohort;
	return base_dt,cohort;