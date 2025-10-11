import os
import pandas as pd

from util import Info

def process_metrics(metrics_folder, dataset_name, info):
    metrics_folder = sorted(metrics_folder)
    
    for idx, path in enumerate(metrics_folder):
        metrics_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        for metric_file in metrics_files:
            metric_df = pd.read_csv(os.path.join(path, metric_file))
            metric_df = metric_df[['timestamp'] + info.metric_names]
            # add 16 hours to the timestamp column
            metric_df['timestamp'] = metric_df['timestamp'] + 16 * 60 * 60
            save_path = f'./parsed_data/{dataset_name}/metrics{idx}/{metric_file}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            metric_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    root_path = '../../datasets'
    for dataset_name in ['SN', 'TT']:
        
        if dataset_name == 'SN':
            dataset_path = os.path.join(root_path, 'SN Dataset', 'data')
            info = Info('socialnetwork')
        elif dataset_name == 'TT':
            dataset_path = os.path.join(root_path, 'TT Dataset', 'data')
            info = Info('TrainTicket')
    
        # list all folders in the dataset path
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        metrics_folder = [os.path.join(dataset_path, f, 'metrics') for f in folders]
        
        process_metrics(metrics_folder, dataset_name, info)