import pandas as pd
import numpy as np
from matrix_completion_copy import matrix_completion
import os

os.chdir("/Users/aaran/MHA_Upload_my_changes/Duke_Dataset")

cols = ["AGE_G", "GENDER", "RACE_G", "ACS", "CHFSEV", "PRE_CULM_CONDITION", "HXANGINA", "HXCEREB", "HXCHF", "HXCOPD", "HXDIAB", "HXHTN", "HXHYL", "HXMI", "HXSMOKE",	"NUMPRMI", "DIASBP_R", "PULSE_R", "SYSBP_R", "CBRUITS",	"HEIGHT_R",	"S3", "WEIGHT_R", "CREATININE_R", "HDL_R", "LDL_R",	"TOTCHOL_R", "CATHAPPR", "DIAGCATH", "INTVCATH", "CORDOM", "GRAFTST", "LADST", "LCXST", "LMST", "LVEF_R", "NUMDZV", "PRXLADST", "RCAST"]

df = pd.read_csv('dukecathr_classification_FeatureSelections_for_Imputations.csv', usecols=cols)

X = df.to_numpy()

X_hat = matrix_completion(X, 50, 1)

df_new = pd.DataFrame(X_hat, columns=cols)

# df_new.to_csv('completed_matrix.csv')