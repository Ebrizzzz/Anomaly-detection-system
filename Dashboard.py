import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import os

# --- Configuration ---
LOG_FILE = "anomaly_log.csv"
FURNACES_CONFIG = {
    "furnace1": ["zone1", "zone2"],
    "furnace2": ["zone3", "zone4", "zone5", "zone6"]
}
ZONE_TCS_CONFIG = { 
    "zone1": ["UgnZon1TempRegAr_TC1", "UgnZon1TempSkyddAr_TC2", "UgnZon1TempVaggOverBandAr_TC3", "UgnZon1TempVaggUnderBandAr_TC4"],
    "zone2": ["UgnZon2TempAr_TC1", "UgnZon2TempSkyddAr_TC2", "UgnZon2TempVaggOverBandAr_TC3", "UgnZon2TempVaggUnderBandAr_TC4"],
    "zone3": ["UgnZon3TempRegAr_TC1", "UgnZon3TempSkyddAr_TC2", "UgnZon3TempVaggAr_TC3", "UgnZon3Temp_TC4_Ar", "UgnZon3Temp_TC5_Ar"],
    "zone4": ["UgnZon4TempAr_TC1", "UgnZon4TempSkyddAr_TC2", "UgnZon4TempVaggAr_TC3", "UgnZon4TempVaggAr_TC4"],
    "zone5": ["UgnZon5TempAr_TC1", "UgnZon5TempSkyddAr_TC2", "UgnZon5TempVaggAr_TC3"],
    "zone6": ["UgnZon6TempAr_TC1", "UgnZon6TempSkyddAr_TC2", "UgnZon6TempVaggAr_TC3", "UgnZon6TempUtgValvAr_TC5"]
}

# --- Helper function ---
# This is used by the Drill-Down page
def create_bar_chart(x_values, y_values, thresholds, title, y_axis_title="Error", colors=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_values, y=y_values, name=y_axis_title, marker_color=colors if colors else 'blue'))
    if thresholds is not None and len(thresholds) == len(x_values):
        for i, x_val in enumerate(x_values):
            if not pd.isna(thresholds[i]) and thresholds[i] is not None:
                fig.add_shape(type="line", x0=i-0.4, x1=i+0.4, y0=thresholds[i], y1=thresholds[i], line=dict(color="white", width=2, dash="dash"))
    fig.update_layout(title_text=title, yaxis_title=y_axis_title)
    return fig

# --- Data Loading ---
@st.cache_data(ttl=10)
def load_log_data(log_file_path):
    if not os.path.exists(log_file_path): return pd.DataFrame()
    try:
        df = pd.read_csv(log_file_path, dtype=str, keep_default_na=False, na_values=['', 'None', 'NaN', 'NULL'])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values(by='timestamp', ascending=False)
        for col in df.columns:
            if any(k in col.lower() for k in ['error', 'threshold', 'actual', 'predicted']): df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'is_anomalous' in col.lower(): df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False, '': False}).fillna(False).astype(bool)
        return df
    except Exception: return pd.DataFrame()

# ==============================================================================
# PAGE 1: SUMMARY ANALYSIS
# ==============================================================================
def render_summary_page(log_df):
    st.header("Historical Summary Analysis")
    st.write("This page provides a high-level overview of anomaly trends over the entire logged period.")

    # --- 1. Overview Anomaly Statistics ---
    st.subheader("Overview Statistics")
    
    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    # CORRECTED Anomaly Frequency Calculation
    anomaly_counts = {}

    # Calculate for furnaces
    for furnace_name in FURNACES_CONFIG.keys():
        column_name = f"furnace_{furnace_name}_is_anomalous"
        if column_name in log_df.columns:
            anomaly_counts[furnace_name] = log_df[column_name].sum()
        else:
            anomaly_counts[furnace_name] = 0

    # Calculate for zones
    all_zones = [zone for zones_list in FURNACES_CONFIG.values() for zone in zones_list]
    for zone_name in all_zones:
        column_name = f"zone_{zone_name}_is_anomalous"
        if column_name in log_df.columns:
            anomaly_counts[zone_name] = log_df[column_name].sum()
        else:
            anomaly_counts[zone_name] = 0


    # Total anomalies
    total_furnace_anomalies = anomaly_counts.get("furnace1", 0) + anomaly_counts.get("furnace2", 0)
    col1.metric("Total Furnace Anomalies", int(total_furnace_anomalies))

    # Most frequent furnace
    if any(anomaly_counts.get(f, 0) > 0 for f in FURNACES_CONFIG.keys()):
        most_frequent_furnace = max(FURNACES_CONFIG.keys(), key=lambda f: anomaly_counts.get(f, 0))
        most_frequent_furnace_count = int(anomaly_counts.get(most_frequent_furnace, 0))
        col2.metric("Most Problematic Furnace", most_frequent_furnace.capitalize(), f"{most_frequent_furnace_count} anomalies")
    else:
        col2.metric("Most Problematic Furnace", "None", "0 anomalies")

    # Most frequent zone
    if any(anomaly_counts.get(z, 0) > 0 for z in all_zones):
        most_frequent_zone = max(all_zones, key=lambda z: anomaly_counts.get(z, 0))
        most_frequent_zone_count = int(anomaly_counts.get(most_frequent_zone, 0))
        col3.metric("Most Problematic Zone", most_frequent_zone.capitalize(), f"{most_frequent_zone_count} anomalies")
    else:
        col3.metric("Most Problematic Zone", "None", "0 anomalies")
    


    # ---Anomaly-Free Percentage ---
    total_timestamps = len(log_df)
    if total_timestamps > 0:
        anomaly_free_timestamps = total_timestamps - total_furnace_anomalies
        anomaly_free_percentage = (anomaly_free_timestamps / total_timestamps) * 100
        col2.metric("Anomaly-Free Uptime", f"{anomaly_free_percentage:.2f}%")
    else:
        col2.metric("Anomaly-Free Uptime", "N/A")




    # --- 2. Time Series Plot of Reconstruction Errors for both furnaces together ---
    st.subheader("Historical Trend Analysis")
    log_df_sorted = log_df.sort_values(by='timestamp', ascending=True)

    fig_trend = go.Figure()
    for furnace in FURNACES_CONFIG.keys():
        fig_trend.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[f'furnace_{furnace}_AE_error'], mode='lines', name=f'{furnace.capitalize()} Error'))
        fig_trend.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[f'furnace_{furnace}_AE_threshold'], mode='lines', name=f'{furnace.capitalize()} Threshold', line=dict(dash='dash')))
    st.plotly_chart(fig_trend, use_container_width=True)

    log_df_sorted = log_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    

    # --- Furnace Time Series Plots with Contribution Analysis ---
    st.markdown("#### Furnace AE Error Over Time")

    for furnace_name in FURNACES_CONFIG.keys():
        st.markdown(f"**{furnace_name.capitalize()}**")
        
        error_col = f'furnace_{furnace_name}_AE_error'
        threshold_col = f'furnace_{furnace_name}_AE_threshold'
        
        if error_col not in log_df_sorted.columns or threshold_col not in log_df_sorted.columns:
            st.warning(f"Data for {furnace_name.capitalize()} not found in log file.")
            continue

        # --- 1. Time Series Plot ---
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[error_col], mode='lines', name='AE Error', line=dict(color='blue')))
        fig_ts.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[threshold_col], mode='lines', name='Threshold', line=dict(color='white', dash='dash')))
        
        anomalous_points = log_df_sorted[log_df_sorted[error_col] > log_df_sorted[threshold_col]]
        if not anomalous_points.empty:
            fig_ts.add_trace(go.Scatter(x=anomalous_points['timestamp'], y=anomalous_points[error_col], mode='markers', name='Anomaly', marker=dict(color='red', size=8, symbol='x')))
        
        fig_ts.update_layout(title=f"AE Reconstruction Error for {furnace_name.capitalize()}", xaxis_title="Timestamp", yaxis_title="Reconstruction Error", showlegend=True)
        st.plotly_chart(fig_ts, use_container_width=True)

        # --- 2. Aggregate and Plot Top Feature Contributions by Error Magnitude(sum of errors) ---
        if not anomalous_points.empty:
            st.markdown(f"**Most Impactful Feature Contributors for all {furnace_name.capitalize()} Anomalies**")

            # This dictionary will store the SUM of errors for each feature.
            contribution_errors = {}

            # Loop through the top 5 contributor columns for this furnace
            for i in range(1, 6):
                name_col = f'furnace_{furnace_name}_contrib{i}_name'
                error_col = f'furnace_{furnace_name}_contrib{i}_error'
                
                
                if name_col in anomalous_points.columns and error_col in anomalous_points.columns:
                    
                    
                    error_sum_by_feature = anomalous_points.groupby(name_col)[error_col].sum()
                    
                    
                    for feature_name, total_error in error_sum_by_feature.items():
                        if feature_name and feature_name != "": 
                            contribution_errors[feature_name] = contribution_errors.get(feature_name, 0) + total_error
            
            if contribution_errors:
                
                contrib_df = pd.DataFrame(list(contribution_errors.items()), columns=['Feature', 'Total_Error'])
                contrib_df = contrib_df.sort_values(by='Total_Error', ascending=False).head(10) # Get top 10 overall by impact

                fig_contrib = go.Figure(go.Bar(
                    x=contrib_df['Feature'],
                    y=contrib_df['Total_Error'],
                    marker_color='indigo'
                ))
                fig_contrib.update_layout(
                    title=f"Total Error Contribution by Feature during {furnace_name.capitalize()} Anomalies",
                    xaxis_title="Feature Name",
                    yaxis_title="Sum of Squared Reconstruction Errors"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
            else:
                st.info(f"No feature contribution data was logged for {furnace_name.capitalize()} anomalies.")
        else:
            st.success(f"No anomalies were detected for {furnace_name.capitalize()} in the selected period.")
    


    # --- Zone Time Series Plots ---
    st.markdown("#### Zone AE Error Over Time")

    all_zones = sorted([zone for zones in FURNACES_CONFIG.values() for zone in zones])

    for zone_name in all_zones:
        st.markdown(f"**{zone_name.capitalize()}**")
        
        error_col = f'zone_{zone_name}_AE_error'
        threshold_col = f'zone_{zone_name}_AE_threshold'
        
        if error_col not in log_df_sorted.columns or threshold_col not in log_df_sorted.columns:
            st.warning(f"Data for {zone_name.capitalize()} not found in log file.")
            continue
            
        fig = go.Figure()

        # Plot main error line
        fig.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[error_col], mode='lines', name='AE Error', line=dict(color='green')))
        
        # Plot threshold line
        fig.add_trace(go.Scatter(x=log_df_sorted['timestamp'], y=log_df_sorted[threshold_col], mode='lines', name='Threshold', line=dict(color='white', dash='dash')))
        
        # Highlight anomalous points in red
        anomalous_points = log_df_sorted[log_df_sorted[error_col] > log_df_sorted[threshold_col]]
        if not anomalous_points.empty:
            fig.add_trace(go.Scatter(x=anomalous_points['timestamp'], y=anomalous_points[error_col], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))

        fig.update_layout(title=f"AE Reconstruction Error for {zone_name.capitalize()}", xaxis_title="Timestamp", yaxis_title="Reconstruction Error", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- 2. Aggregate and Plot Top Feature Contributions by Error Magnitude(sum of errors) ---
        if not anomalous_points.empty:
            st.markdown(f"**Most Impactful Feature Contributors for all {zone_name.capitalize()} Anomalies**")

            # This dictionary will store the SUM of errors for each feature.
            contribution_errors = {}

            # Loop through the top 5 contributor columns for this zone
            for i in range(1, 6):
                name_col = f'zone_{zone_name}_contrib{i}_name'
                error_col = f'zone_{zone_name}_contrib{i}_error'
                
                if name_col in anomalous_points.columns and error_col in anomalous_points.columns:
                    
                    
                    error_sum_by_feature = anomalous_points.groupby(name_col)[error_col].sum()
                    
                    for feature_name, total_error in error_sum_by_feature.items():
                        if feature_name and feature_name != "": 
                            contribution_errors[feature_name] = contribution_errors.get(feature_name, 0) + total_error
            
            if contribution_errors:
                contrib_df = pd.DataFrame(list(contribution_errors.items()), columns=['Feature', 'Total_Error'])
                contrib_df = contrib_df.sort_values(by='Total_Error', ascending=False).head(10) # Get top 10 overall by impact

                fig_contrib = go.Figure(go.Bar(
                    x=contrib_df['Feature'],
                    y=contrib_df['Total_Error'],
                    marker_color='indigo'
                ))
                fig_contrib.update_layout(
                    title=f"Total Error Contribution by Feature during {zone_name.capitalize()} Anomalies",
                    xaxis_title="Feature Name",
                    yaxis_title="Sum of Squared Reconstruction Errors"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
            else:
                st.info(f"No feature contribution data was logged for {zone_name.capitalize()} anomalies.")
        else:
            st.success(f"No anomalies were detected for {zone_name.capitalize()} in the selected period.")


    
   
    
    # --- 3. Anomaly Frequency Bar Chart ---
    st.subheader("Anomaly Frequency by Zone")
    zone_names = [z for zones in FURNACES_CONFIG.values() for z in zones]
    zone_counts = [int(log_df.get(f"zone_{z}_is_anomalous", pd.Series(False)).sum()) for z in zone_names]
    
    fig_freq = go.Figure(go.Bar(
        x=zone_names, 
        y=zone_counts,
        text=zone_counts,  
        textposition='auto' 
    ))
    fig_freq.update_layout(title="Total Anomaly Count per Zone", yaxis_title="Number of Anomalous Events")
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # --- 4. Top N Anomalous Events Table by Furnaces---
    st.subheader("Most Severe Anomalous Events by Furnace")
    anomalous_events_df = log_df[log_df['furnace_furnace1_is_anomalous'] | log_df['furnace_furnace2_is_anomalous']].copy()
    anomalous_events_df['max_error'] = anomalous_events_df[['furnace_furnace1_AE_error', 'furnace_furnace2_AE_error']].max(axis=1)
    top_events = anomalous_events_df.sort_values(by='max_error', ascending=False).head(10)
    st.dataframe(top_events[['timestamp', 'max_error', 'furnace_furnace1_is_anomalous', 'furnace_furnace2_is_anomalous']])

    # --- 5. Top N Anomalous Events Table by Zones ---
    st.subheader("Most Severe Anomalous Events by Zone")
    zone_anomalous_events = []
    for zone in all_zones:  
        zone_col = f'zone_{zone}_is_anomalous'
        if zone_col in log_df.columns:
            zone_events = log_df[log_df[zone_col]].copy()
            zone_events['max_error'] = zone_events[f'zone_{zone}_AE_error']
            top_zone_events = zone_events.sort_values(by='max_error', ascending=False).head(10)
            top_zone_events['zone'] = zone.capitalize()
            zone_anomalous_events.append(top_zone_events[['timestamp', 'max_error', 'zone']])
    if zone_anomalous_events:
        zone_anomalous_events_df = pd.concat(zone_anomalous_events, ignore_index=True)
        st.dataframe(zone_anomalous_events_df.sort_values(by='max_error', ascending=False))
        


# ==============================================================================
# PAGE 2: DRILL-DOWN ANALYSIS
# ==============================================================================
def render_drill_down_page(log_df):
    # --- Sidebar for selecting data view ---
    view_mode = st.sidebar.radio("Select Data Source:", ("Latest Real-time", "Historical Timestamp"))
    selected_row_data = None

    if view_mode == "Latest Real-time":
        st.sidebar.write("Showing the most recent entry.")
        if not log_df.empty: selected_row_data = log_df.iloc[0]
        if st.sidebar.button("Refresh Latest Data"): st.cache_data.clear(); st.rerun()
    
    elif view_mode == "Historical Timestamp":
        if not log_df.empty:
            timestamps = log_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()
            if timestamps:
                ts_str = st.sidebar.selectbox("Select Timestamp:", timestamps)
                selected_row_data = log_df[log_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') == ts_str].iloc[0]

    # --- Displaying the charts based on selected_row_data ---
    if selected_row_data is not None:
        st.header(f"Analysis for: {selected_row_data['timestamp']}")

        # ==============================================================================
        # 1. FURNACE LEVEL ANALYSIS
        # ==============================================================================
        st.subheader("Furnace Level Analysis")
        furnace_names_display = list(FURNACES_CONFIG.keys())
        furnace_errors = [selected_row_data.get(f"furnace_{f}_AE_error", np.nan) for f in furnace_names_display]
        furnace_thresholds = [selected_row_data.get(f"furnace_{f}_AE_threshold", np.nan) for f in furnace_names_display]
        furnace_anomalous_flags = [selected_row_data.get(f"furnace_{f}_is_anomalous", False) for f in furnace_names_display]
        furnace_colors = ['red' if flag else 'lightgreen' for flag in furnace_anomalous_flags]

        # Display the main furnace error chart
        if any(not pd.isna(e) for e in furnace_errors):
            fig_furnace = create_bar_chart(furnace_names_display, furnace_errors, furnace_thresholds, "Furnace AE Errors", colors=furnace_colors)
            st.plotly_chart(fig_furnace, use_container_width=True)
        else:
            st.write("No furnace AE error data available for this timestamp.")
            
        # ---Display Feature Contributions for any anomalous furnace ---
        for i, furnace_name in enumerate(furnace_names_display):
            if furnace_anomalous_flags[i]:
                st.markdown(f"#### Top Contributing Features for **{furnace_name.capitalize()}** Anomaly")
                
                contrib_names = []
                contrib_errors = []
                for j in range(1, 6):  # Check for top 5 contributors
                    name_col = f"furnace_{furnace_name}_contrib{j}_name"
                    error_col = f"furnace_{furnace_name}_contrib{j}_error"
                    if name_col in selected_row_data and pd.notna(selected_row_data[name_col]) and selected_row_data[name_col] != "":
                        contrib_names.append(selected_row_data[name_col])
                        contrib_errors.append(selected_row_data.get(error_col, 0))
                
                if contrib_names:
                    fig_contrib = go.Figure(go.Bar(
                        y=contrib_names,
                        x=contrib_errors,
                        orientation='h',
                        marker_color='orange'
                    ))
                    fig_contrib.update_layout(
                        title_text=f'Feature Contributions to {furnace_name.capitalize()} AE Error',
                        xaxis_title='Squared Reconstruction Error',
                        yaxis_title='Feature Name',
                        yaxis={'categoryorder':'total ascending'}  # Show largest bar at the top
                    )
                    st.plotly_chart(fig_contrib, use_container_width=True)

        # ==============================================================================
        # 2. ZONE AND THERMOCOUPLE LEVEL ANALYSIS (HIERARCHICAL DRILL-DOWN)
        # ==============================================================================
        for furnace_name, zones_in_furnace in FURNACES_CONFIG.items():
            is_furnace_anomalous = selected_row_data.get(f"furnace_{furnace_name}_is_anomalous", False)
            
            # Only show zone/TC details if the parent furnace was flagged as anomalous
            if is_furnace_anomalous:
                st.subheader(f"Zone Level Drill-Down for Anomalous {furnace_name.capitalize()}")
                
                # --- Prepare Zone Data ---
                zone_names_display = zones_in_furnace
                zone_errors = [selected_row_data.get(f"zone_{z}_AE_error", np.nan) for z in zone_names_display]
                zone_thresholds = [selected_row_data.get(f"zone_{z}_AE_threshold", np.nan) for z in zone_names_display]
                zone_anomalous_flags = [selected_row_data.get(f"zone_{z}_is_anomalous", False) for z in zone_names_display]
                zone_colors = ['red' if flag else 'lightgreen' for flag in zone_anomalous_flags]

                if any(not pd.isna(e) for e in zone_errors):
                    fig_zone = create_bar_chart(zone_names_display, zone_errors, zone_thresholds, f"{furnace_name.capitalize()} - Zone AE Errors", colors=zone_colors)
                    st.plotly_chart(fig_zone, use_container_width=True)

                    # --- NEW: Display Feature Contributions for any anomalous zones ---
                    for k, zone_name in enumerate(zone_names_display):
                        if zone_anomalous_flags[k]:
                            st.markdown(f"#### Top Contributing Features for **{zone_name.capitalize()}** Anomaly")
                            
                            contrib_names_zone = []
                            contrib_errors_zone = []
                            for j in range(1, 6):
                                name_col = f"zone_{zone_name}_contrib{j}_name"
                                error_col = f"zone_{zone_name}_contrib{j}_error"
                                if name_col in selected_row_data and pd.notna(selected_row_data[name_col]) and selected_row_data[name_col] != "":
                                    contrib_names_zone.append(selected_row_data[name_col])
                                    contrib_errors_zone.append(selected_row_data.get(error_col, 0))
                            
                            if contrib_names_zone:
                                fig_zone_contrib = go.Figure(go.Bar(y=contrib_names_zone, x=contrib_errors_zone, orientation='h', marker_color='coral'))
                                fig_zone_contrib.update_layout(title_text=f'Feature Contributions to {zone_name.capitalize()} AE Error', xaxis_title='Squared Reconstruction Error', yaxis_title='Feature Name', yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig_zone_contrib, use_container_width=True)

                    # --- TC Level Drill-Down ---
                    anomalous_zone_found_for_tc = False
                    for k, zone_name in enumerate(zone_names_display):
                        if zone_anomalous_flags[k]:
                            if not anomalous_zone_found_for_tc:
                                st.subheader(f"Thermocouple Level Drill-Down for Anomalous Zones")
                                anomalous_zone_found_for_tc = True

                            st.markdown(f"#### Regression Analysis for **{zone_name.capitalize()}**")
                            tcs_in_zone = ZONE_TCS_CONFIG.get(zone_name, [])
                            tc_actuals = [selected_row_data.get(f"tc_{zone_name}_{tc}_actual", np.nan) for tc in tcs_in_zone]
                            tc_predicts = [selected_row_data.get(f"tc_{zone_name}_{tc}_predicted", np.nan) for tc in tcs_in_zone]
                            
                            plot_tc_names = []
                            plot_tc_data_actual = []
                            plot_tc_data_predicted = []

                            for j, tc_name in enumerate(tcs_in_zone):
                                if pd.notna(tc_actuals[j]) and pd.notna(tc_predicts[j]):
                                    plot_tc_names.append(tc_name)
                                    plot_tc_data_actual.append(tc_actuals[j])
                                    plot_tc_data_predicted.append(tc_predicts[j])
                            
                            if plot_tc_names:
                                fig_tc = go.Figure()
                                fig_tc.add_trace(go.Bar(
                                    name='Actual', 
                                    x=plot_tc_names, 
                                    y=plot_tc_data_actual, 
                                    marker_color='skyblue',
                                    text=[f"{val:.1f}" for val in plot_tc_data_actual],
                                    textposition='auto'
                                ))
                                fig_tc.add_trace(go.Bar(
                                    name='Predicted', 
                                    x=plot_tc_names, 
                                    y=plot_tc_data_predicted, 
                                    marker_color='royalblue',
                                    text=[f"{val:.1f}" for val in plot_tc_data_predicted],
                                    textposition='auto'
                                ))
                                fig_tc.update_layout(barmode='group', title_text=f"{zone_name.capitalize()} - Predicted vs. Actual TC Temperatures", yaxis_title="Temperature (°C)")
                                st.plotly_chart(fig_tc, use_container_width=True)

    else:
        if view_mode == "Latest Real-time":
            st.info("Waiting for new data in the log file...")


# ==============================================================================
# MAIN APP ROUTER
# ==============================================================================
def main():
    st.set_page_config(layout="wide")
    st.title("Furnace Anomaly Detection Dashboard")

    log_df = load_log_data(LOG_FILE)

    if log_df.empty:
        st.warning("No anomaly log data found. Please ensure the analysis script is running.")
        st.stop()
    
    # ---Page selection in the sidebar ---
    page = st.sidebar.selectbox("Select Dashboard Page", ["Drill-Down Analysis", "Summary Analysis"])
    
    if page == "Drill-Down Analysis":
        render_drill_down_page(log_df)
    elif page == "Summary Analysis":
        render_summary_page(log_df)

if __name__ == "__main__":
    main()