# salary-prediction

## File List

Clone the repo to your local machine to make sure files are in the correct location.

All files coded on Python 3.7.3

Salary Prediction Problem.pdf is a presentation of the project results. This is a good starting point to get a feeling of the overall project structure.

salary-prediction-pipeline-full.py is a python script that takes in training and test data and runs the full machine learning pipeline including training, model selection, and prediction. The results are then saved into an SQL database. 

*IMPORTANT* This script will not run unless you update logincreds.py with information for an SQL database. For obvious reasons, I have replaced my SQL username/password with dummies.

First Portfolio Project.ipynb is a jupyter notebook that runs through my entire data science process from EDA through model validation.

EDAhelperfunctions.py is a module of helper functions for EDA.

clean_helperfunctions.py is a module of helper functions for data cleaning.

logincreds.py is a way to store SQL credentials. Must be updated with your credentials and stored in the same folder as First Portfolio Project.ipynb and salary-prediction-pipeline-full.py. I have done this for convenience, if in a professional environment I would use a secure way of sending credentials such as hashicorp vault.

requirements.txt - list of dependencies and versions.

data folder contains the data used in the analysis.
