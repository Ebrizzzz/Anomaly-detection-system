import os
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime
import time

# ---------------------------
# Configurations
# ---------------------------
MODEL_BASE = r"zone_models_final" # path to autoencoder models
TC_MODEL_BASE = r"tc_regression_models_final_withlag" # path to regression models for TCs
LOG_FILE = "anomaly_log.csv"

FURNACES = {
    "furnace1": ["zone1", "zone2"],
    "furnace2": [ "zone3", "zone4", "zone5", "zone6"]
}

ZONE_TCS = {
    "zone1": ["UgnZon1TempRegAr_TC1", "UgnZon1TempSkyddAr_TC2", "UgnZon1TempVaggOverBandAr_TC3", "UgnZon1TempVaggUnderBandAr_TC4"],
    "zone2": ["UgnZon2TempAr_TC1", "UgnZon2TempSkyddAr_TC2", "UgnZon2TempVaggOverBandAr_TC3", "UgnZon2TempVaggUnderBandAr_TC4"],
    "zone3": ["UgnZon3TempRegAr_TC1", "UgnZon3TempSkyddAr_TC2", "UgnZon3TempVaggAr_TC3", "UgnZon3Temp_TC4_Ar", "UgnZon3Temp_TC5_Ar"],
    "zone4": ["UgnZon4TempAr_TC1", "UgnZon4TempSkyddAr_TC2", "UgnZon4TempVaggAr_TC3", "UgnZon4TempVaggAr_TC4"],
    "zone5": ["UgnZon5TempAr_TC1", "UgnZon5TempSkyddAr_TC2", "UgnZon5TempVaggAr_TC3"],
    "zone6": ["UgnZon6TempAr_TC1", "UgnZon6TempSkyddAr_TC2", "UgnZon6TempVaggAr_TC3", "UgnZon6TempUtgValvAr_TC5"]
}

ZONE_COLUMNS = {
    "zone1": ['AvgTemp_ZON1_M', 'UgnZon1BransleFlodeAr_Under', 'UgnZon1BransleFlodeAr_Over',
             'UgnZon1OljaFlodeAr_FT131', 'UgnZon1TempRegAr_TC1', 'UgnZon1TempSkyddAr_TC2',
             'UgnZon1TempVaggOverBandAr_TC3', 'UgnZon1TempVaggUnderBandAr_TC4', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON1_M_lag1', 'UgnZon1BransleFlodeAr_Under_lag1', 'UgnZon1BransleFlodeAr_Over_lag1',
             'UgnZon1OljaFlodeAr_FT131_lag1', 'UgnZon1TempRegAr_TC1_lag1', 'UgnZon1TempSkyddAr_TC2_lag1',
             'UgnZon1TempVaggOverBandAr_TC3_lag1', 'UgnZon1TempVaggUnderBandAr_TC4_lag1', 'LineControlHastSverk4_1Act_lag1'],
    "zone2": ['AvgTemp_ZON2_M', 'UgnZon2BransleFlodeAr_Under', 'UgnZon2BransleFlodeAr_Over',
             'UgnZon2OljaFlodeAr_FT231', 'UgnZon2TempAr_TC1', 'UgnZon2TempSkyddAr_TC2',
             'UgnZon2TempVaggOverBandAr_TC3', 'UgnZon2TempVaggUnderBandAr_TC4', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON2_M_lag1', 'UgnZon2BransleFlodeAr_Under_lag1', 'UgnZon2BransleFlodeAr_Over_lag1',
             'UgnZon2OljaFlodeAr_FT231_lag1', 'UgnZon2TempAr_TC1_lag1', 'UgnZon2TempSkyddAr_TC2_lag1',
             'UgnZon2TempVaggOverBandAr_TC3_lag1', 'UgnZon2TempVaggUnderBandAr_TC4_lag1', 'LineControlHastSverk4_1Act_lag1'],
    "zone3": ['AvgTemp_ZON3_M', 'UgnZon3BransleFlodeAr_Under', 'UgnZon3BransleFlodeAr_Over',
             'UgnZon3OljaFlodeAr_FT331', 'UgnZon3TempRegAr_TC1', 'UgnZon3TempSkyddAr_TC2',
             'UgnZon3TempVaggAr_TC3', 'UgnZon3Temp_TC4_Ar', 'UgnZon3Temp_TC5_Ar', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON3_M_lag1', 'UgnZon3BransleFlodeAr_Under_lag1', 'UgnZon3BransleFlodeAr_Over_lag1',
             'UgnZon3OljaFlodeAr_FT331_lag1', 'UgnZon3TempRegAr_TC1_lag1', 'UgnZon3TempSkyddAr_TC2_lag1',
             'UgnZon3TempVaggAr_TC3_lag1', 'UgnZon3Temp_TC4_Ar_lag1', 'UgnZon3Temp_TC5_Ar_lag1', 'LineControlHastSverk4_1Act_lag1'],
    "zone4": ['AvgTemp_ZON4_M', 'UgnZon4BransleFlodeAr_Under', 'UgnZon4BransleFlodeAr_Over',
             'UgnZon4OljaFlodeAr_FT431', 'UgnZon4TempAr_TC1', 'UgnZon4TempSkyddAr_TC2',
             'UgnZon4TempVaggAr_TC3', 'UgnZon4TempVaggAr_TC4', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON4_M_lag1', 'UgnZon4BransleFlodeAr_Under_lag1', 'UgnZon4BransleFlodeAr_Over_lag1',
             'UgnZon4OljaFlodeAr_FT431_lag1', 'UgnZon4TempAr_TC1_lag1', 'UgnZon4TempSkyddAr_TC2_lag1',
             'UgnZon4TempVaggAr_TC3_lag1', 'UgnZon4TempVaggAr_TC4_lag1', 'LineControlHastSverk4_1Act_lag1'],
    "zone5": ['AvgTemp_ZON5_M', 'UgnZon5OljaFlodeAr_FT531', 'UgnZon5TempAr_TC1', 'UgnZon5TempSkyddAr_TC2',
             'UgnZon5TempVaggAr_TC3', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON5_M_lag1', 'UgnZon5OljaFlodeAr_FT531_lag1', 'UgnZon5TempAr_TC1_lag1', 'UgnZon5TempSkyddAr_TC2_lag1',
             'UgnZon5TempVaggAr_TC3_lag1', 'LineControlHastSverk4_1Act_lag1'],
    "zone6": ['AvgTemp_ZON6_M', 'UgnZon6OljaFlodeAr_FT631', 'UgnZon6TempAr_TC1', 'UgnZon6TempSkyddAr_TC2',
             'UgnZon6TempVaggAr_TC3', 'UgnZon6TempUtgValvAr_TC5', 'LineControlHastSverk4_1Act',
             'AvgTemp_ZON6_M_lag1', 'UgnZon6OljaFlodeAr_FT631_lag1', 'UgnZon6TempAr_TC1_lag1', 'UgnZon6TempSkyddAr_TC2_lag1',
             'UgnZon6TempVaggAr_TC3_lag1', 'UgnZon6TempUtgValvAr_TC5_lag1', 'LineControlHastSverk4_1Act_lag1']
}

FURNACE_COLUMNS = {
    "furnace1": list(dict.fromkeys(ZONE_COLUMNS["zone1"] + ZONE_COLUMNS["zone2"])),  
    "furnace2": list(dict.fromkeys(
        ZONE_COLUMNS["zone3"] + ZONE_COLUMNS["zone4"] + ZONE_COLUMNS["zone5"] + ZONE_COLUMNS["zone6"]))
}


# ---------------------------
# Load utilities
# ---------------------------
def load_scaler(path):
    return joblib.load(path)

def load_autoencoder(path):
    return load_model(path)

def load_threshold(path):
    return np.load(path, allow_pickle=True).item()

def load_regression_model(path):
    return joblib.load(path)

# ---------------------------
# Inference functions
# ---------------------------
def get_reconstruction_error(model, scaler, row):
    X = scaler.transform([row])
    X_pred = model.predict(X)
    return np.mean(np.square(X - X_pred))

def is_anomalous(error, threshold):
    return error > threshold

def predict_tc_and_error_single_row(model, scaler, input_features_np_array, actual_value): 


    input_scaled = scaler.transform([input_features_np_array]) 
    predicted = model.predict(input_scaled)[0]
    error = abs(predicted - actual_value)
    return predicted, error
def get_feature_contribution(model, scaler, row_np, feature_names):
    """
    Calculates the reconstruction error for each individual feature.
    Returns a dictionary of feature_name: error.
    """
    X = scaler.transform([row_np])
    X_pred = model.predict(X, verbose=0)
    
    # Calculate the squared error for each feature individually
    feature_errors = np.square(X - X_pred)[0] 
    
    # Create a dictionary mapping feature names to their errors
    contribution_dict = dict(zip(feature_names, feature_errors))
    
    # Sort the dictionary by error in descending order
    sorted_contributions = sorted(contribution_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_contributions

# ---------------------------
# Main Row Analyzer
# ---------------------------
def analyze_row(row: pd.Series, loaded_models_bundle): 
    log = {
        "timestamp": row["DateTime"],
        "furnace_level": {},
        "zone_level": {},
        "tc_level": {}
    }

    # --- Access pre-loaded models ---
    ae_models = loaded_models_bundle["autoencoders"]
    ae_scalers = loaded_models_bundle["ae_scalers"]
    ae_thresholds = loaded_models_bundle["ae_thresholds"]
    reg_models = loaded_models_bundle["reg_models"]
    reg_scalers = loaded_models_bundle["reg_scalers"]


    for furnace, zones in FURNACES.items():
        if furnace not in ae_models:
            print(f"Skipping furnace {furnace}: model/scaler/threshold not pre-loaded.")
            continue

        # Prepare data for furnace autoencoder
        furnace_feature_names = FURNACE_COLUMNS[furnace]
        if row[furnace_feature_names].isnull().any():
            print(f"NaNs detected in data for {furnace} autoencoder at {row['DateTime']}. Assigning high error.")
            f_error = np.inf 
        else:
            furnace_data_np = row[furnace_feature_names].values.astype(float)
            f_error = get_reconstruction_error(ae_models[furnace], ae_scalers[furnace], furnace_data_np)

        log["furnace_level"][furnace] = f_error

        if is_anomalous(f_error, ae_thresholds[furnace]):
            print(f"  ANOMALY in {furnace} (Error: {f_error:.4f} > Thresh: {ae_thresholds[furnace]:.4f})")

            # --- Get Feature Contributions for the Furnace Anomaly ---
            contributions = get_feature_contribution(ae_models[furnace], ae_scalers[furnace], furnace_data_np, furnace_feature_names)
            print(f"    Top 5 contributing features for {furnace} anomaly:")
            for feature, error in contributions[:5]:
                print(f"      - {feature}: {error:.4f}")
            
            log[f'furnace_{furnace}_contributions'] = contributions


            zone_errors_info = {} # Store more info for zones
            for zone in zones:
                if zone not in ae_models:
                    print(f"Skipping zone {zone}: model/scaler/threshold not pre-loaded.")
                    continue

                zone_feature_names = ZONE_COLUMNS[zone]
                 # Check for NaNs in the required features for THIS zone model
                if row[zone_feature_names].isnull().any():
                    print(f"NaNs detected in data for {zone} autoencoder at {row['DateTime']}. Assigning high error.")
                    z_error = np.inf
                else:
                    zone_data_np = row[zone_feature_names].values.astype(float)
                    z_error = get_reconstruction_error(ae_models[zone], ae_scalers[zone], zone_data_np)

                zone_errors_info[zone] = {
                    "error": z_error,
                    "threshold": ae_thresholds[zone],
                    "is_anomalous": is_anomalous(z_error, ae_thresholds[zone])
                }
                if zone_errors_info[zone]["is_anomalous"]:
                    print(f"    ANOMALY in {zone} (Error: {z_error:.4f} > Thresh: {ae_thresholds[zone]:.4f})")
                    # ---Get Feature Contributions for the Zone Anomaly ---
                    zone_contributions = get_feature_contribution(ae_models[zone], ae_scalers[zone], zone_data_np, zone_feature_names)
                    print(f"      Top 5 contributing features for {zone} anomaly:")
                    for feature, error in zone_contributions[:5]:
                        print(f"        - {feature}: {error:.4f}")
                    
                    # Add to log
                    log[f'zone_{zone}_contributions'] = zone_contributions
                    

            log["zone_level"][furnace] = {
                z: info["error"] for z, info in zone_errors_info.items()
            }
            
            anomalous_zones_in_furnace = [z for z, info in zone_errors_info.items() if info["is_anomalous"]]
            
            if not anomalous_zones_in_furnace:
                print(f"  Furnace {furnace} flagged, but no specific zone exceeded its AE threshold. No TC drill-down.")
                continue # No specific zone to drill into for TC regression

            
            # --- This block processes ALL anomalous zones ---
            if anomalous_zones_in_furnace:
                print(f"    Anomalous zones found in {furnace}: {', '.join(anomalous_zones_in_furnace)}. Running TC regression for each.")

                for zone_to_process in anomalous_zones_in_furnace:
                    print(f"      --- Analyzing TCs for anomalous zone: {zone_to_process} ---")
                    
                    # ------------- TC REGRESSION SECTION -------------
                    if zone_to_process in ZONE_TCS and zone_to_process in reg_models:
                        for tc in ZONE_TCS[zone_to_process]:
                            # Ensure TC model and scaler were loaded
                            if tc not in reg_models[zone_to_process] or tc not in reg_scalers[zone_to_process]:
                                print(f"        Skipping TC {tc} in {zone_to_process}: regression model/scaler not pre-loaded.")
                                continue

                            # Prepare features for THIS TC's regression model
                            regression_input_feature_names = ZONE_COLUMNS[zone_to_process].copy()
                            if tc in regression_input_feature_names:
                                regression_input_feature_names.remove(tc)
                            else:
                                print(f"        Warning: Target TC {tc} not found in ZONE_COLUMNS for {zone_to_process}. Skipping TC regression.")
                                continue

                            actual_val = row[tc]

                            if row[regression_input_feature_names].isnull().any() or pd.isnull(actual_val):
                                print(f"        NaNs detected for TC {tc} regression. Skipping prediction.")
                                log["tc_level"][f"{zone_to_process}_{tc}"] = {"predicted": None, "actual": actual_val, "error": None, "status": "NaN in input"}
                                continue

                            tc_input_data_np = row[regression_input_feature_names].values.astype(float)

                            predicted, error = predict_tc_and_error_single_row(
                                reg_models[zone_to_process][tc],
                                reg_scalers[zone_to_process][tc],
                                tc_input_data_np,
                                actual_val
                            )
                            log["tc_level"][f"{zone_to_process}_{tc}"] = {"predicted": predicted, "actual": actual_val, "error": error}
                            print(f"        TC {tc}: Actual={actual_val:.2f}, Pred={predicted:.2f}, Err={error:.2f}")

            else: # anomalous_furnace_name but no specific zone triggered its threshold
                 log["zone_level"][furnace] = {z : info["error"] for z, info in zone_errors_info.items()} # Log all zone errors

        else: # Furnace not anomalous
            print(f"  {furnace} is NORMAL (Error: {f_error:.4f} <= Thresh: {ae_thresholds[furnace]:.4f})")


    return log

# ---------------------------
# Logging Function
# ---------------------------
def log_results(log_dict, loaded_models_bundle):
    """
    Takes the nested log dictionary from analyze_row, flattens it into a single-row DataFrame
    with a consistent structure, and appends it to the CSV log file to then be used by streamlit application.
    """
    
    # ---------------------------
    # 1. INITIALIZE ALL POSSIBLE COLUMNS
    # ---------------------------    
    flat_log = {"timestamp": log_dict.get("timestamp")}
    TOP_N_CONTRIBUTORS = 5

    # Create a unique, sorted list of all configured zone names for consistent ordering
    all_configured_zones = sorted(list(set(zone for zones in FURNACES.values() for zone in zones)))

    # Initialize columns for Furnaces (AE error, threshold, flag, and top N contributors)
    for furnace_name in FURNACES.keys():
        flat_log[f"furnace_{furnace_name}_AE_error"] = np.nan
        flat_log[f"furnace_{furnace_name}_AE_threshold"] = np.nan
        flat_log[f"furnace_{furnace_name}_is_anomalous"] = False
        for i in range(TOP_N_CONTRIBUTORS):
            flat_log[f"furnace_{furnace_name}_contrib{i+1}_name"] = ""
            flat_log[f"furnace_{furnace_name}_contrib{i+1}_error"] = np.nan

    # Initialize columns for Zones
    for zone_name in all_configured_zones:
        flat_log[f"zone_{zone_name}_AE_error"] = np.nan
        flat_log[f"zone_{zone_name}_AE_threshold"] = np.nan
        flat_log[f"zone_{zone_name}_is_anomalous"] = False
        for i in range(TOP_N_CONTRIBUTORS):
            flat_log[f"zone_{zone_name}_contrib{i+1}_name"] = ""
            flat_log[f"zone_{zone_name}_contrib{i+1}_error"] = np.nan

    # Initialize columns for Thermocouples
    for zone_name, tc_list in ZONE_TCS.items():
        for tc_name in tc_list:
            key = f"{zone_name}_{tc_name}"
            flat_log[f"tc_{key}_predicted"] = np.nan
            flat_log[f"tc_{key}_actual"] = np.nan
            flat_log[f"tc_{key}_error"] = np.nan
            flat_log[f"tc_{key}_status"] = ""

    # ---------------------------
    # 2. POPULATE THE FLATTENED LOG WITH DATA FROM THE CURRENT ROW
    # ---------------------------

    # Populate Furnace data
    for furnace_name, error in log_dict.get("furnace_level", {}).items():
        threshold = loaded_models_bundle["ae_thresholds"].get(furnace_name, np.nan)
        flat_log[f"furnace_{furnace_name}_AE_error"] = error
        flat_log[f"furnace_{furnace_name}_AE_threshold"] = threshold
        if not pd.isna(error) and not pd.isna(threshold):
            flat_log[f"furnace_{furnace_name}_is_anomalous"] = error > threshold

    # Populate Zone data
    for furnace_context, zones_data in log_dict.get("zone_level", {}).items():
        for zone_name, error in zones_data.items():
            threshold = loaded_models_bundle["ae_thresholds"].get(zone_name, np.nan)
            flat_log[f"zone_{zone_name}_AE_error"] = error
            flat_log[f"zone_{zone_name}_AE_threshold"] = threshold
            if not pd.isna(error) and not pd.isna(threshold):
                flat_log[f"zone_{zone_name}_is_anomalous"] = error > threshold

    # Populate Thermocouple data
    for tc_key, tc_data in log_dict.get("tc_level", {}).items():
        flat_log[f"tc_{tc_key}_predicted"] = tc_data.get("predicted", np.nan)
        flat_log[f"tc_{tc_key}_actual"] = tc_data.get("actual", np.nan)
        flat_log[f"tc_{tc_key}_error"] = tc_data.get("error", np.nan)
        flat_log[f"tc_{tc_key}_status"] = tc_data.get("status", "")
        
    # --- Populate Feature Contribution data ---
    for key, contributions in log_dict.items():
        if key.endswith("_contributions"):
            entity_full_name = key.replace("_contributions", "")
            
            for i, (feature_name, error_val) in enumerate(contributions[:TOP_N_CONTRIBUTORS]):
                flat_log[f"{entity_full_name}_contrib{i+1}_name"] = feature_name
                flat_log[f"{entity_full_name}_contrib{i+1}_error"] = error_val

    # ---------------------------
    # 3. DEFINE COLUMN ORDER AND SAVE TO CSV
    # ---------------------------
    
    ordered_columns = ["timestamp"]
    for furnace_name in FURNACES.keys():
        ordered_columns.extend([
            f"furnace_{furnace_name}_AE_error",
            f"furnace_{furnace_name}_AE_threshold",
            f"furnace_{furnace_name}_is_anomalous"
        ])
        for i in range(TOP_N_CONTRIBUTORS):
             ordered_columns.extend([
                f"furnace_{furnace_name}_contrib{i+1}_name",
                f"furnace_{furnace_name}_contrib{i+1}_error"
            ])

    for zone_name in all_configured_zones:
        ordered_columns.extend([
            f"zone_{zone_name}_AE_error",
            f"zone_{zone_name}_AE_threshold",
            f"zone_{zone_name}_is_anomalous"
        ])
        for i in range(TOP_N_CONTRIBUTORS):
            ordered_columns.extend([
                f"zone_{zone_name}_contrib{i+1}_name",
                f"zone_{zone_name}_contrib{i+1}_error"
            ])
            
    for zone_name, tc_list in ZONE_TCS.items():
        for tc_name in tc_list:
            key = f"{zone_name}_{tc_name}"
            ordered_columns.extend([
                f"tc_{key}_predicted",
                f"tc_{key}_actual",
                f"tc_{key}_error",
                f"tc_{key}_status"
            ])

    # Create a DataFrame from the flat_log dictionary using the specified order
    df_log = pd.DataFrame([flat_log])
    df_log = df_log[ordered_columns] 

    # Append to CSV file
    file_exists = os.path.exists(LOG_FILE)
    df_log.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)


if __name__ == "__main__":
    print("Loading models and scalers, please wait...")
    # --- Pre-load all models, scalers, thresholds ---
    loaded_models_bundle = {
        "autoencoders": {}, "ae_scalers": {}, "ae_thresholds": {},
        "reg_models": {}, "reg_scalers": {}
    }

    FURNACE_AND_ZONE_NAMES = list(FURNACES.keys()) + [zone for zones_list in FURNACES.values() for zone in zones_list]
    for entity_name in FURNACE_AND_ZONE_NAMES:
        try:
            loaded_models_bundle["autoencoders"][entity_name] = load_autoencoder(os.path.join(MODEL_BASE, f"{entity_name}_autoencoder.keras"))
            loaded_models_bundle["ae_scalers"][entity_name] = load_scaler(os.path.join(MODEL_BASE, f"{entity_name}_scaler.pkl"))
            loaded_models_bundle["ae_thresholds"][entity_name] = load_threshold(os.path.join(MODEL_BASE, f"{entity_name}_threshold.npy"))
            print(f"  Successfully loaded AE for {entity_name}")
        except Exception as e:
            print(f"  ERROR loading AE for {entity_name}: {e}")

    for zone_name_key, tc_list in ZONE_TCS.items(): # zone_name_key is "zone1", "zone2" etc.
        zone_path_prefix = zone_name_key.upper()
        # replace ZONE with ZON in the path
        zone_path_prefix = zone_path_prefix.replace("ZONE", "ZON")
        loaded_models_bundle["reg_models"][zone_name_key] = {}
        loaded_models_bundle["reg_scalers"][zone_name_key] = {}
        for tc_name in tc_list:
            try:
                
                model_path = os.path.join(TC_MODEL_BASE, f"{zone_path_prefix}_{tc_name}_ridge_regressor.pkl")
                scaler_path = os.path.join(TC_MODEL_BASE, f"{zone_path_prefix}_{tc_name}_ridge_scaler.pkl")
                loaded_models_bundle["reg_models"][zone_name_key][tc_name] = load_regression_model(model_path)
                loaded_models_bundle["reg_scalers"][zone_name_key][tc_name] = load_scaler(scaler_path)
                print(f"  Successfully loaded REG for {zone_name_key} - {tc_name}")
            except Exception as e:
                print(f"  ERROR loading REG for {zone_name_key} - {tc_name}: {e}")
    print("Model loading complete.")

    # --- Load data ---
    
    data = pd.read_parquet("ThermocoupleData_2025-04_v2.parquet") # data with the format used normally in operations

    data = data.sort_values(by='DateTime').reset_index(drop=True)

    print(f"\nStarting analysis for {len(data)} rows...")
    limit = 20000 # Or however many rows to process
    previous_row = None

    processed_count = 0
    for _, row in data.iterrows():
        if previous_row is None:
            previous_row = row
            print(f"Initializing with first row: {row['DateTime']}. No analysis performed.")
            continue

        print(f"\nProcessing row for timestamp: {row['DateTime']}")

        combined_features = {}

        # Add current features
        for col in row.index:
            if col != 'DateTime': # Exclude the timestamp itself from features
                combined_features[col] = row[col]
        # Add lagged features from the previous row
        for col in previous_row.index:
            if col != 'DateTime':
                combined_features[f"{col}_lag1"] = previous_row[col]

        analysis_input_row = pd.Series(combined_features)
        analysis_input_row['DateTime'] = row['DateTime']


        log = analyze_row(analysis_input_row, loaded_models_bundle)
        log_results(log, loaded_models_bundle)

        previous_row = row


        processed_count += 1
        
        if processed_count >= limit:
            print(f"Reached processing limit of {limit} rows.")
            break
            
        if processed_count < len(data) and processed_count < limit:
            print(f"Waiting n seconds before processing next row...")
            #time.sleep(n)
            
    print("Analysis finished.")