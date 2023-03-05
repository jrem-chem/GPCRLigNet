# GPCRLigNet

Contains python scripts described by the methods in the paper "GPCRLigNet: Rapid Screening for GPCR Active Ligands Using Machine Learning"

# How to make predictions with GPCRLigNet

1) edit the following file to your needs for generating fingerprints like the trainingset
    druglikeness_analysis/generate_fpts.py
2) edit the following file to turn those fingerprints into logit activity predictions. You will need to set the best performing model path in the python script
    druglikeness_analysis/make_predictions.py
    best model is found at models/cicular_4_models_6_17_21/model_cicular4.tf

Unfortunately the full datasets/models were too large for github, if you would like to see anything else please dont hesitate to reachout.
