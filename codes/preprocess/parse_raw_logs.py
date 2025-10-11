import os
import json
from transformers import BertTokenizer, BertModel
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import numpy as np
import pandas as pd
import re


'''
parse_raw_logs.py 是一个用于处理日志文件的 Python 脚本，主要功能是提取日志模板并将其保存为 JSON 文件和 CSV 文件。具体来说：
日志模板提取：使用 drain3 库对无故障的日志数据进行模板化处理，生成一组日志模板。
日志时间戳提取：通过正则表达式从日志中提取时间戳，并将其转换为 Unix 时间格式。
故障日志处理：将有故障的日志与提取的模板匹配，生成包含时间戳、服务名称和事件模板的 CSV 文件。
最终输出结果存储在 ./parsed_data/{dataset_name} 目录下，包括：
templates.json：存储提取的日志模板。
多个 logs{idx}.csv 文件：存储每个子文件夹中的日志信息。
'''

def extract_unix_timestamp(log: str) -> float:

    time_pattern = r"\d{4}-(?:[A-Za-z]{3}|\d{2})-\d{2} \d{2}:\d{2}:\d{2}\.\d+"

    match = re.search(time_pattern, log)
    if not match:
        return None

    time_str = match.group(0)

    try:
        unix_time = pd.to_datetime(time_str, format="%Y-%b-%d %H:%M:%S.%f").timestamp()

    except ValueError:
        unix_time = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S.%f").timestamp()


    unix_time += 8 * 3600

    return unix_time



'''
定义一个函数 extract_log_template，接收三个参数：
fault_free_dataset_path：无故障日志数据的路径。
fault_time_dataset_path：有故障日志数据的路径。
dataset_name：数据集名称（如 'SN' 或 'TT'）。
'''
def extract_log_template(fault_free_dataset_path, fault_time_dataset_path, dataset_name):

    subfolders = [f for f in os.listdir(fault_free_dataset_path) if os.path.isdir(os.path.join(fault_free_dataset_path, f))]

    log_paths = [os.path.join(fault_free_dataset_path, f, 'logs.json') for f in subfolders]

    all_log_strs = []

    for path in log_paths:
        with open(path, 'r') as f:
            logs = json.load(f)

        for k, v in logs.items():
            all_log_strs.extend(v)


    config = TemplateMinerConfig()
    config.load('./drain3.ini')
    config.profiling_enabled = True
    miner = TemplateMiner(config=config)
    for log_str in all_log_strs:
        miner.add_log_message(log_str)
    templates = []
    for cluster in miner.drain.clusters:
        templates.append(cluster.get_template())
        print(cluster)
    print("*"*90)
    templates_save_path = os.path.join("./parsed_data", dataset_name)
    os.makedirs(templates_save_path, exist_ok=True)
    
    templates_file_path = os.path.join(templates_save_path, "templates.json")
    with open(templates_file_path, 'w') as f:
        json.dump(templates, f)



    subfolders = [f for f in os.listdir(fault_time_dataset_path) if os.path.isdir(os.path.join(fault_time_dataset_path, f))]
    log_paths = [os.path.join(fault_time_dataset_path, f, 'logs.json') for f in subfolders]

    log_paths = sorted(log_paths)
    for idx, path in enumerate(log_paths):
        with open(path, 'r') as f:
            log_dict = json.load(f)

        df = {'timestamp':[], 'service':[], 'events':[]}
        for service, log_list in log_dict.items():
            for log_line in log_list:
                match = miner.match(log_line)
                if match:
                    log_temp = match.get_template()
                else:
                    print(log_line)
                    log_temp = "Unseen"
                log_time = extract_unix_timestamp(log_line)
                assert log_time is not None, log_line
                # print(log_temp, log_time) ;exit()
                df['timestamp'].append(log_time)
                df['service'].append(service)
                df['events'].append(log_temp)

        df = pd.DataFrame(df)
        df.to_csv(os.path.join("./parsed_data", dataset_name, "logs"+str(idx)+".csv"))
                
    print("======"*30)
    


if __name__ == '__main__':
    root_path = '../../datasets'
    for dataset_name in ['SN', 'TT']:
        
        if dataset_name == 'SN':
            fault_free_dataset_path = os.path.join(root_path, 'SN Dataset', 'no fault')
            fault_time_dataset_path = os.path.join(root_path, 'SN Dataset', 'data')
        elif dataset_name == 'TT':
            fault_free_dataset_path = os.path.join(root_path, 'TT Dataset', 'no fault')
            fault_time_dataset_path = os.path.join(root_path, 'TT Dataset', 'data')
    
        extract_log_template(fault_free_dataset_path, fault_time_dataset_path, dataset_name)
