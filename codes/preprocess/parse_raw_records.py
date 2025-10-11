import os
import re
import json
import argparse

def SN_service_names(name):
    mapping = {
        'socialnetwork-text-service-1': 'text-service',
        'socialnetwork-home-timeline-service-1': 'home-timeline-service',
        'socialnetwork-media-service-1': 'media-service',
        'socialnetwork-post-storage-service-1':'post-storage-service',
        'socialnetwork-social-graph-service-1':'social-graph-service',
        'socialnetwork-url-shorten-service-1':'url-shorten-service',
        'socialnetwork-nginx-thrift-1':'nginx-web-server',
        'socialnetwork-unique-id-service-1':'unique-id-service',
        'socialnetwork-user-service-1':'user-service',
        'socialnetwork-compose-post-service-1':'compose-post-service',
        'socialnetwork-user-timeline-service-1':'user-timeline-service',
        'socialnetwork-user-mention-service-1':'user-mention-service'
    }
    if name in mapping:
        return mapping[name]
    else:
        raise KeyError('Service name not found: {}'.format(name))

def TT_service_names(name):
    return name.split("_")[1]


def process_raw_records(dataset_path, data_name):
    # 对指定路径下的原始记录文件进行处理，并生成新的 JSON 文件。
    path = os.path.join(dataset_path, 'data')
    # 从 dataset_path/data 目录中读取所有以 .json 结尾的文件名，并按字母顺序排序。
    record_files = [f for f in os.listdir(path) if f.endswith('.json')]
    record_files.sort()
    # 创建一个字典 record2idx，将每个文件名与其索引关联。
    # enumerate(record_files)
    # record_files 是一个列表，包含以 .json 结尾的文件名（按字母顺序排序）。
    # enumerate(record_files) 会返回一个迭代器，生成每项的索引和对应的文件名，例如：(0, 'file1.json'), (1, 'file2.json')。
    # 字典推导式 {file_name: idx for idx, file_name in enumerate(record_files)}
    # 这是一个字典推导式，用于生成一个新的字典。
    # 对于 enumerate(record_files) 返回的每一项 (idx, file_name)，将 file_name 作为键，idx 作为值存入字典。
    # 最终生成的字典形式为：{'file1.json': 0, 'file2.json': 1, ...}。
    # 用途
    # 该字典 record2idx 的目的是为每个文件名分配一个唯一的索引值，方便后续处理中通过文件名快速获取其对应的索引
    record2idx = {file_name: idx for idx, file_name in enumerate(record_files)}
    # 对于每个文件，读取其内容（JSON 格式），并提取以下信息
    for file_name in record_files:
        with open(os.path.join(path, file_name), 'r') as f:
            # 读取其内容（JSON 格式）
            records = json.load(f)
        # 创建一个名为 processed_records 的空字典，用于存储处理后的记录。
        processed_records = {}

        processed_records['start'] = int(records['start'])
        processed_records['end'] = int(records['end'])
        # add 16 hours to the start and end time
        # to align with the time interal in the file-name
        # https://github.com/BEbillionaireUSD/Eadro/issues/11#issuecomment-1887219310
        # 为开始和结束时间添加16小时，以与文件名中的时间间隔对齐
        processed_records['start'] += 16 * 3600
        processed_records['end'] += 16 * 3600
        # 为处理记录的字典添加一个名为 faults 的空列表，用于存储故障信息。
        processed_records['faults'] = []
        for fault in records['faults']:
            # 读取解析出的json中每条fault的信息，包括微服务的名称、故障类型、开始时间和持续时间。
            service = SN_service_names(fault['name']) if data_name == 'SN' else TT_service_names(fault['name'])
            fault_type = fault['fault']
            start = int(fault['start'])
            end = int(fault['start'] + fault['duration'])
            
            start += 16 * 3600
            end += 16 * 3600

            # 在列表中添加字典，填入刚刚解析的信息
            processed_records['faults'].append({
                'service': service,
                'fault_type': fault_type,
                's': start,
                'e': end
            })

        # 构建保存路径并确保目录存在，str(record2idx[file_name])提取的是record2idx这个字典中文件名对应的索引值，作为文件名的一部分。
        save_path = os.path.join('./parsed_data', data_name, 'records' + str(record2idx[file_name]) + '.json')
        # os.makedirs 的作用
        # os.makedirs 用于递归地创建目录，即如果指定路径中的父目录不存在，也会一并创建。
        # 例如：os.makedirs('./parsed_data/SN') 会创建 parsed_data 和 SN 目录（如果它们尚不存在）。
        # exist_ok 参数的作用
        # 默认情况下，如果目标目录已经存在，os.makedirs 会抛出 FileExistsError 异常。
        # 当设置 exist_ok=True 时，如果目标目录已经存在，则不会抛出异常，而是静默跳过创建操作
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 将处理后的记录字典写入创建的文件
        with open(save_path, 'w') as f:
            json.dump(processed_records, f)
        
        # # check the timestamps as well as the datetime in file names. 
        # import pandas as pd
        # start_datetime = pd.to_datetime(processed_records['start'], unit='s')
        # end_datetime = pd.to_datetime(processed_records['end'], unit='s')
        # print(f"{file_name}: \n {start_datetime} -- {end_datetime}")
    
    
    


if __name__ == "__main__":
    root_path = '../../datasets'
    for dataset_name in ['SN', 'TT']:
        if dataset_name == 'SN':
            dataset_path = os.path.join(root_path, 'SN Dataset')
        elif dataset_name == 'TT':
            dataset_path = os.path.join(root_path, 'TT Dataset')
    
        process_raw_records(dataset_path, dataset_name)
    
