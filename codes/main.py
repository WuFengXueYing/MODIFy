from torch.utils.data import Dataset, DataLoader
import torch
import dgl
class chunkDataset(Dataset): #[node_num, T, else]
    """
           初始化函数，用于构建图数据结构并存储相关数据。

           Args:
               chunks (dict): 包含多个数据块的字典，每个数据块包含日志、指标、追踪信息以及对应的错误标签。
               node_num (int): 图中节点的数量。
               edges (tuple): 图的边信息，包含两个列表，分别表示边的源节点和目标节点。
           """
    def __init__(self, chunks, node_num, edges):
        # 存储图数据及其对应的错误标签
        self.data = []
        # 用于将索引映射到数据块ID的字典
        self.idx2id = {}
        # 遍历chunks字典，构建图数据结构并存储相关信息
        for idx, chunk_id in enumerate(chunks.keys()):
            # 将索引映射到数据块ID
            self.idx2id[idx] = chunk_id
            chunk = chunks[chunk_id]
            # 使用DGL库创建有向图，并设置节点特征，edges[0] 和 edges[1] 分别表示图中的源节点和目标节点。0->1
            graph = dgl.graph((edges[0], edges[1]), num_nodes=node_num)
            # 设置节点的日志特征，graph.ndata 是一个字典，用于存储图中节点的特征数据。每个键对应一个特征名称，值是一个张量（tensor），表示所有节点在该特征上的值
            # torch.FloatTensor 是 PyTorch 中的一个函数，用于将输入数据转换为浮点型张量（tensor）。
            graph.ndata["logs"] = torch.FloatTensor(chunk["logs"])
            # 设置节点的指标特征
            graph.ndata["metrics"] = torch.FloatTensor(chunk["metrics"])
            # 设置节点的追踪特征
            graph.ndata["traces"] = torch.FloatTensor(chunk["traces"])
            # 将图及其对应的错误节点存储到data列表中
            # 如果 chunk["culprit"] 为 -1，表示该数据块中没有故障节点。
            # 否则，chunk["culprit"] 表示故障节点的索引（从 0 开始）
            # 这样做的目的是将每个数据块的图结构和对应的标签组合在一起，形成一个完整的数据项，方便后续的数据加载和处理。
            self.data.append((graph, chunk["culprit"]))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __get_chunk_id__(self, idx):
        return self.idx2id[idx]

from utils import *
from base import BaseModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", default=42, type=int)

### Training params
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epoches", default=70, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument("--node_feat_dim", default=64, type=int)

##### Fuse params
parser.add_argument("--self_attn", default=True, type=lambda x: x.lower() == "true")
# parser.add_argument("--self_attn", default=False, type=lambda x: x.lower() == "tru e")
parser.add_argument("--fuse_dim", default=128, type=int)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--beta", default=0.1, type=float)
parser.add_argument("--locate_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--detect_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--detector_rank", default=16, type=int)
parser.add_argument("--locator_rank", default=16, type=int)

##### Source params
parser.add_argument("--log_dim", default=16, type=int)
parser.add_argument("--trace_kernel_sizes", default=[2], type=int, nargs='+')
parser.add_argument("--trace_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--metric_kernel_sizes", default=[2], type=int, nargs='+')
parser.add_argument("--metric_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--graph_hiddens", default=[64], type=int, nargs='+')
parser.add_argument("--attn_head", default=4, type=int, help="For gat or gat-v2")
parser.add_argument("--activation", default=0.2, type=float, help="use LeakyReLU, shoule be in (0,1)")


##### Data params
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--result_dir", default="../result/")

### add_module
parser.add_argument("--use_transformer", default=True , type=lambda x: x.lower() == "true", help="Use TransformerEncoder for TraceModel and MetricModel")
parser.add_argument("--use_CGLU", default=True, type=lambda x: x.lower() == "true", help="Use CGLU for MultsourceEncoder")
parser.add_argument("--use_TraceDifussion", default=True, type=lambda x: x.lower() == "true", help="Use CGLU for GraphModel")


params = vars(parser.parse_args())

import logging
def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")
    

def collate(data):

    graphs, labels = map(list, zip(*data))
    batched_graph = dgl.batch(graphs)
    return batched_graph , torch.tensor(labels)

def run(evaluation_epoch=10):
    data_dir = os.path.join("./chunks", params["data"])
    metadata = read_json(os.path.join(data_dir, "metadata.json"))
    event_num, node_num, metric_num =  metadata["event_num"], metadata["node_num"], metadata["metric_num"]
    edges = metadata["edges"]
    params["chunk_lenth"] = metadata["chunk_lenth"]


    if params["seq_len"] is None:
        params["seq_len"] = params["chunk_lenth"]

    hash_id = dump_params(params)
    params["hash_id"] = hash_id
    seed_everything(params["random_seed"])
    device = get_device(params["gpu"])

    train_chunks, test_chunks = load_chunks(data_dir)
    train_data = chunkDataset(train_chunks, node_num, edges)
    test_data = chunkDataset(test_chunks, node_num, edges)
    graph_example = train_data[0][0]  # 获取第一个图样本
    feature_keys = ['logs', 'metrics', 'traces']
    total_node_feat_dim = sum([
        graph_example.ndata[key].view(graph_example.ndata[key].size(0), -1).size(1)  # 展平后计算维度
        for key in feature_keys
    ])
    params["node_feat_dim"] = total_node_feat_dim  # 设置总特征维度


    train_dl = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
    model = BaseModel(event_num, metric_num, node_num, device, **params)
    scores, converge = model.fit(train_dl, test_dl, evaluation_epoch=evaluation_epoch)
    module_info = f"Transformer: {params['use_transformer']}, " \
                  f"TraceDifussion: {params['use_TraceDifussion']}, " \
                  f"CGLU: {params['use_CGLU']} "


    dump_scores(params["result_dir"], hash_id, scores, converge, params["data"], params["epoches"], params["lr"], params["gpu"], module_info)
    logging.info("Current hash_id {}".format(hash_id))

if "__main__" == __name__:
    run()
