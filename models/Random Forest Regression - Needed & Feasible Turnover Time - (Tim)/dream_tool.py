import PySimpleGUI as sg
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import math

# Load the trained RandomForestRegressor model
joblib_file = "rf_feas.pkl"
rf_realized = joblib.load(joblib_file)

# Load the residuals
rf_realized_train_residuals = np.load("rf_feas_residuals.npy")

# Load the trained RandomForestRegressor model
joblib_file = "rf_needed.pkl"
rf_needed = joblib.load(joblib_file)

# Load the residuals
rf_needed_train_residuals = np.load("rf_needed_residuals.npy")

# Function to calculate dynamic percentile
def predict_with_dynamic_percentile(X_test, model, residuals, percentile):
    # Predict using the loaded model
    y_pred = model.predict(X_test)
    
    # Calculate the specified percentile of the residuals
    residual_percentile = np.percentile(residuals, percentile)
    
    # Adjust the predicted means by the calculated percentile of the residuals
    y_pred_adjusted = y_pred + residual_percentile
    
    return y_pred, y_pred_adjusted
    
def predict_turnover(Station, Rollingstock, Driver_change, Combine, Split, Number_of_carriages, Date_and_time, Percentile, Acceptable_delay):
    columns = ['NUMBER_CARRIAGES', 'DRIVER_CHANGE', 'COMBINE', 'SPLIT', 'DAY_OF_WEEK_sin', 'DAY_OF_WEEK_cos', 'HOUR_sin', 'HOUR_cos', 'STATION_Ah', 'STATION_Akm', 'STATION_Alm', 'STATION_Almo', 'STATION_Amf', 'STATION_Amfs', 'STATION_Aml', 'STATION_Amr', 'STATION_Ana', 'STATION_Apd', 'STATION_Apn', 'STATION_Asb', 'STATION_Asd', 'STATION_Asdl', 'STATION_Asdm', 'STATION_Asdz', 'STATION_Asn', 'STATION_Ass', 'STATION_Avat', 'STATION_Bd', 'STATION_Bdg', 'STATION_Bgn', 'STATION_Bhv', 'STATION_Bkl', 'STATION_Bl', 'STATION_Brn', 'STATION_Bsk', 'STATION_Bsmz', 'STATION_Btl', 'STATION_Bv', 'STATION_Db', 'STATION_Ddr', 'STATION_Dld', 'STATION_Dmnz', 'STATION_Dn', 'STATION_Dr', 'STATION_Dron', 'STATION_Dt', 'STATION_Dtcp', 'STATION_Dv', 'STATION_Dvd', 'STATION_Ed', 'STATION_Ehv', 'STATION_Ekz', 'STATION_Es', 'STATION_Gd', 'STATION_Gdg', 'STATION_Gdm', 'STATION_Gerp', 'STATION_Gn', 'STATION_Gs', 'STATION_Gv', 'STATION_Gvc', 'STATION_Gvm', 'STATION_Gvmw', 'STATION_Gz', 'STATION_Hd', 'STATION_Hde', 'STATION_Hdr', 'STATION_Hdrz', 'STATION_Hfd', 'STATION_Hgl', 'STATION_Hgv', 'STATION_Hks', 'STATION_Hlm', 'STATION_Hm', 'STATION_Hn', 'STATION_Hnk', 'STATION_Hr', 'STATION_Hrl', 'STATION_Hrt', 'STATION_Ht', 'STATION_Htn', 'STATION_Htnc', 'STATION_Hvs', 'STATION_Hwd', 'STATION_Krg', 'STATION_Laa', 'STATION_Ldl', 'STATION_Ledn', 'STATION_Lis', 'STATION_Lls', 'STATION_Lw', 'STATION_Mas', 'STATION_Mdb', 'STATION_Mp', 'STATION_Mt', 'STATION_Ndb', 'STATION_Nkk', 'STATION_Nm', 'STATION_Ns', 'STATION_O', 'STATION_Odb', 'STATION_Ohze', 'STATION_Ost', 'STATION_Ozbm', 'STATION_Pmo', 'STATION_Pt', 'STATION_Rai', 'STATION_Rlb', 'STATION_Rm', 'STATION_Rsd', 'STATION_Rsn', 'STATION_Rtd', 'STATION_Rtn', 'STATION_Rtst', 'STATION_Rtz', 'STATION_Sdm', 'STATION_Sgn', 'STATION_Shl', 'STATION_Sptn', 'STATION_St', 'STATION_Std', 'STATION_Swk', 'STATION_Tb', 'STATION_Tbu', 'STATION_Tl', 'STATION_Ut', 'STATION_Utg', 'STATION_Uto', 'STATION_Vl', 'STATION_Vndc', 'STATION_Vs', 'STATION_Vtn', 'STATION_Wc', 'STATION_Wd', 'STATION_Wdn', 'STATION_Wf', 'STATION_Wm', 'STATION_Wp', 'STATION_Wt', 'STATION_Wv', 'STATION_Ypb', 'STATION_Zd', 'STATION_Zl', 'STATION_Zlw', 'STATION_Zp', 'STATION_Ztmo', 'STATION_Zvt', 'STATION_Zwd', 'ROLLINGSTOCK_TYPE_DDZ', 'ROLLINGSTOCK_TYPE_FLIRT', 'ROLLINGSTOCK_TYPE_ICM', 'ROLLINGSTOCK_TYPE_SLT', 'ROLLINGSTOCK_TYPE_SNG', 'ROLLINGSTOCK_TYPE_VIRM']
    df = pd.DataFrame([[0]*len(columns)], columns=columns)

    df["STATION_"+Station] = 1
    df["ROLLINGSTOCK_TYPE_"+Rollingstock] = 1
    df["DRIVER_CHANGE"] = Driver_change
    df["NUMBER_CARRIAGES"] = Number_of_carriages
    df["COMBINE"] = Combine
    df["SPLIT"] = Split
    datetime_obj = pd.to_datetime(Date_and_time)
    day_of_week = datetime_obj.dayofweek
    hour = datetime_obj.hour
    df["DAY_OF_WEEK_sin"] = math.sin(2 * math.pi * day_of_week / 7)
    df["DAY_OF_WEEK_cos"] = math.cos(2 * math.pi * day_of_week / 7)
    df["HOUR_sin"] = math.sin(2 * math.pi * hour / 24)
    df["HOUR_cos"] = math.cos(2 * math.pi * hour / 24)

    y_pred_real, y_pred_real_adjusted = predict_with_dynamic_percentile(df, rf_realized, rf_realized_train_residuals, Percentile)
    y_pred_needed, y_pred_needed_adjusted = predict_with_dynamic_percentile(df, rf_needed, rf_needed_train_residuals, Percentile)
    y_pred_needed = y_pred_needed + Acceptable_delay
    y_pred_needed_adjusted = y_pred_needed_adjusted + Acceptable_delay


    return y_pred_real, y_pred_real_adjusted, y_pred_needed, y_pred_needed_adjusted

    #print(f"For an acceptable delay of {Acceptable_delay} seconds, the average turnover time needed is:", round(y_pred_needed[0]), "seconds.")
    #print(f"For an acceptable delay of {Acceptable_delay} seconds, the {Percentile}th percentile turnover time needed is:", round(y_pred_needed_adjusted[0]), "seconds.")
    
    #print()
    
    #print("Average realized turnover time:", round(y_pred_real[0]), "seconds.")
    #print(f"{Percentile}th percentile realized turnover time:", round(y_pred_real_adjusted[0]), "seconds.")


layout = [
    [sg.Text("Station")],
    [sg.InputText()],
    [sg.Text("Type of Rolling Stock")],
    [sg.InputText()],
    [sg.Text("Driver Change")],
    [sg.InputText()],
    [sg.Text("Combine")],
    [sg.InputText()],
    [sg.Text("Split")],
    [sg.InputText()],
    [sg.Text("Number of Carriages")],
    [sg.InputText()],
    [sg.Text("Date and Time")],
    [sg.InputText()],
    [sg.Text("Percentage of Trains within Delay")],
    [sg.InputText()],
    [sg.Text("Acceptable delay (seconds)")],
    [sg.InputText()],
    [sg.Button('Calculate Turnover Time')],
    [sg.Multiline(size=(70, 10), key='-OUTPUT-', disabled=True)]
]

window = sg.Window('Turnover Time Calculator', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event == 'Calculate Turnover Time':
        Station = values[0]
        Rollingstock = values[1]
        Driver_change = int(values[2])
        Combine = int(values[3])
        Split = int(values[4])
        Number_of_carriages = int(values[5])
        Date_and_time = values[6]
        Percentile = int(values[7])
        Acceptable_delay = int(values[8])
        
        y_pred_real, y_pred_real_adjusted, y_pred_needed, y_pred_needed_adjusted = predict_turnover(
            Station, Rollingstock, Driver_change, Combine, Split, Number_of_carriages, Date_and_time, Percentile, Acceptable_delay
        )
        
        feasible_string = "not "
        if y_pred_needed_adjusted > y_pred_real_adjusted:
            feasible_string = ""
        
        result_message = (
            f"The turnover time needed for the top {Percentile}% of trains to depart within a delay of {Acceptable_delay} seconds is: {round(y_pred_needed_adjusted[0])} seconds.\n"
            "\n"
            f"The feasible turnover time for {Percentile}% of trains is: {round(y_pred_real_adjusted[0])} seconds.\n"
            "\n"
            f"The needed turnover time is {feasible_string}feasible!"
        )
        
        window['-OUTPUT-'].update(result_message)

window.close()
