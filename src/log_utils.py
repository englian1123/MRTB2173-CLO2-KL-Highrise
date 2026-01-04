import pandas as pd
import os
import datetime

# Centralized log path
# LOG_PATH = 'monitoring_logs.csv'

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 
# Construct the path to the log file
LOG_PATH = os.path.join(script_dir, '..', 'log', 'monitoring_logs.csv')

def log_prediction_feedback(history_entry):
    """
    Refines history entry to match monitoring dashboard expectations
    and appends to the central log file.
    """
    # Mapping app keys to monitor keys
    log_data = {
        'timestamp': datetime.datetime.now(),
        'model_version': 'v1_baseline', # Default to v1, we will call this twice or handle logic
        'prediction': history_entry.get('Prediction_V1'),
        'latency_ms': history_entry.get('Latency_V1_ms'),
        'feedback_score': history_entry.get('Rating_V1'),
        'feedback_text': history_entry.get('Comments'),
        'input_summary': f"Mukim: {history_entry.get('Mukim')}, Scheme: {history_entry.get('SchemeName')}"
    }

    # Log V1
    df_v1 = pd.DataFrame([log_data])
    
    # Log V2
    log_data_v2 = log_data.copy()
    log_data_v2['model_version'] = 'v2_project_specific'
    log_data_v2['prediction'] = history_entry.get('Prediction_V2')
    log_data_v2['latency_ms'] = history_entry.get('Latency_V2_ms')
    log_data_v2['feedback_score'] = history_entry.get('Rating_V2')
    df_v2 = pd.DataFrame([log_data_v2])

    final_df = pd.concat([df_v1, df_v2])
    
    file_exists = os.path.isfile(LOG_PATH)
    final_df.to_csv(LOG_PATH, mode='a', index=False, header=not file_exists)