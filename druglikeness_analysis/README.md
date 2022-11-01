# README

This folder contains python scripts to independently asses the correlation between GPCR activity as predicted by GPCRLigNet and drug-likeness. This task is broken down into the following scripts, which consecutively ran in the following order:

### 1. `generate_ftps.py`

Generates an array of Morgan fingerprints for a given list of mol objects to be assessed. Molecules with more than 80 atoms, non-organic atoms (atoms other than H, C, N, O, F, Na, P, S, Cl, Br, I), or atoms with a net charge larger than one were filtered out and not converted to Morgan fingerprints. The array of fingerprints and corresponding list of SMILE strings were saved for use in the next script.



### 2. `make_predictions.py`

Loads the array of Morgan fingerprints and runs GPCRLigNet to generate activity predictions for each fingerprint. The array of predictions is then saved.



### 3. `frame_data.py`

Loads the array of GPCRLigNet predictions and corresponding array of smile strings created in `generate_fpts.py`. Creates a dataframe containing the SMILE string, GPCR activity prediction, computed descriptors, and drug-likeness metrics for each molecule. The Veber, Ghose, and Lipinski Ro5 metrics are pass/fail tests, and so molecules were classified as such, for each metric. For QED, molecules with a QED > 0.5 are considered drug-like, and those with QED < 0.5 are considered not drug-like. No molecule had a QED value that was exactly 0.5, so this case was ignored.  Saves this dataframe for future analysis.



### 4. `plot_data.py`

Loads the dataframe created in `frame_data.py` and splits data into two groups for each drug-likeness metric (drug-like or not drug-like). For each metric, performs the Kolmogorov-Smirnov test on the predicted GPCR activity for each subgroup. Plots the distribution of GPCR activity for drug-like and non drug-like molecules, respectively, for each metric. 