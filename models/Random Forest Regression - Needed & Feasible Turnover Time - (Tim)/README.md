# Random Forest Regression - Needed & Feasible Turnover Time - by Tim Rentenaar
This folder contains code developed for the UU ADS thesis "Next Stop: Robust Timetables; Predicting NS Train Turnover Times"

## Description
- **Model_train.ipynb:** the notebook used for training and exporting both needed and feasible turnover time models
- **dream_tool.py:** the prototype of NS's dream tool, imports the models and predicts based on user input
- **baseline_models.ipynb:** the baseline models the developed models were compared to: NS's current method, linear regression, random forest quantile regression
- **Random_search.ipynb:** narrowing down the hyperparameter space with randomized search
- **Grid_search.ipynb:** grid search within the narrowed down hyperparameter space to find the best set
- **Exploratory_Data_Analysis.ipynb:** creating exploratory plots

## Files that couldn't be uploaded to GitHub due to size
- **two_model_optimization_test.ipynb:** the notebook testing splitting into multiple regression models and combining them with a classification model to try to optimize performance
- Model pkl files used in the dream tool

These files can be sent over upon request
