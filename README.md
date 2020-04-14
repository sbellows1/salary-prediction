# salary-prediction

## File List

Salary Prediction Problem.pdf is a presentation of the project results. This is a good starting point to get a feeling of the overall project structure.

salary-prediction-pipeline-full.py is a python script that takes in training and test data and runs the full machine learning pipeline including training, model selection, and prediction. The results are then saved into an SQL database. *IMPORTANT* This script will not run unless you update logincreds.py with information for an SQL database. For obvious reasons, I have replaced my SQL username/password with dummies.

First Portfolio Project.ipynb is a jupyter notebook that runs through my entire data science process from EDA through model validation.
*IMPORTANT* The notebook will not fully run unless you update logincreds.py with information for an SQL database. However, you should be able to view the current output. As a side note, this notebook takes a LONG time to run due to optimizing hyperparameters.

EDAhelperfunctions.py is a module of helper functions. Must be stored in the same folder as First Portfolio Project.ipynb and salary-prediction-pipeline-full.py for those to run correctly.

clean_helperfunctions.py is a module of helper functions. Must be stored in the same folder as First Portfolio Project.ipynb and salary-prediction-pipeline-full.py for those to run correctly.

logincreds.py is a way to store SQL credentials. Must be updated with your credentials and stored in the same folder as First Portfolio Project.ipynb and salary-prediction-pipeline-full.py

## Required Libraries
numpy
pandas
matplotlib
seaborn
sklearn
psycopg2
sqlalchemy
