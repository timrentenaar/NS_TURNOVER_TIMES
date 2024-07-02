
# LightGBM Model - Stephan Berende

This folder contains the code for the LightGBM model, used to predict turnover times. The folder has the following contents:

- **LightGBM_default:** a notebook with the default LightGBM trained on 1 year of data. Contains both the non-split and 4-split default models. 
- **LightGBM_optimization:** contains the the script for hyperparameter tuning of the 4-split model.
- **output:** contains the output of the hyperparameter optimization process (parameters, optimization plots, feature importances, optimization history).
- **ThesisNS_StephanBerende:** The actual thesis document itself.

*Note: due to variance in models and data sampling, the given metrics (MAE & RMSE) may be slightly different with each model run, and therefore deviate from the thesis scores as well.*
