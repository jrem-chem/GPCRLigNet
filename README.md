# GPCRLigNet

Contains python scripts described by the methods in the paper "GPCRLigNet: Rapid Screening for GPCR Active Ligands Using Machine Learning" https://link.springer.com/article/10.1007/s10822-023-00497-2

# How to make predictions with GPCRLigNet

1) edit the following file to your needs for generating the morgan fingerprints from smiles strings: 
    druglikeness_analysis/generate_fpts.py
2) edit the following file to turn those fingerprints into logit GPCR activity predictions:
    druglikeness_analysis/make_predictions.py
3) the best performing model was models/cicular_4_models_6_17_21/model_cicular4.tf
4) other models can be found in models/cicular_4_models_6_17_21/

Unfortunately the full datasets were too large for github, if you would like to see anything else please dont hesitate to reachout.
