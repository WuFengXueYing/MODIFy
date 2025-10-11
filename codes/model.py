
import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
from dgl.nn import GlobalAttentionPooling


import math

class GraphModel(nn.Module):
    def __init__(self, in_dim, graph_hiddens=[64, 128], device='cpu', attn_head=4, activation=0.2, **kwargs):
        super(GraphModel, self).__init__()
        layers = []
        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            dropout = kwargs["attn_drop"] if "attn_drop" in kwargs else 0
            layers.append(GATv2Conv(in_feats, out_feats=hidden, num_heads=attn_head,
                                    attn_drop=dropout, negative_slope=activation, allow_zero_in_degree=True))
        self.maxpool = nn.MaxPool1d(attn_head)
        self.net = nn.Sequential(*layers).to(device)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))


    def forward(self, graph, x):

        out = None
        for layer in self.net:
            if out is None: out = x
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out)


from torch import nn
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, max_seq_len=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):  # [batch, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch, input_dim]
        embedded = self.embedding(x)

        output = self.transformer(embedded)
        return output.permute(1, 0, 2)  # [batch, seq_len, d_model]

class Diffusion(nn.Module):
    def __init__(self, node_feat_dim, device, input_dim, output_dim, noise_steps=10 , beta_start=1e-5, beta_end=0.01 ):
        super(Diffusion, self).__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.node_feat_dim = node_feat_dim
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim


        self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

        self.denoiser = Denoiser(in_channels=input_dim, out_channels=output_dim).to(device)
    def forward(self, x):

        batch_size, seq_len, feat_dim = x.shape


        x = x.to(self.device)

        t = torch.randint(0, self.noise_steps, (batch_size,), device=x.device).long()

        alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1)

        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

        denoised_x = self.denoiser(noisy_x.permute(0, 2, 1)).permute(0, 2, 1)

        return denoised_x


class Denoiser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim=768):
        super(MultiScaleDWConv, self).__init__()
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=True, groups=dim)
        # self.dwconv7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=True, groups=dim)
        # self.dwconv9 = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, bias=True, groups=dim)
        self.mca_weights = nn.Parameter(torch.ones(2))
    def forward(self, x, H, W, bz):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, -1, W).contiguous()
        weights = F.softmax(self.mca_weights, dim=0)
        x3 = self.dwconv3(x)
        x5 = self.dwconv5(x)
        # x7 = self.dwconv7(x)
        # x9 = self.dwconv9(x)
        # x = (x3 + x5) / 2
        x = weights[0] * x3 + weights[1] * x5

        x = x.flatten(2).transpose(1, 2)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.size()
        x_pool = self.global_avg_pool(x.transpose(1, 2)).view(B, C)
        x_se = self.sigmoid(self.fc2(self.relu(self.fc1(x_pool)))).view(B, 1, C)
        return x * x_se


class CGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(CGLU, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.se = SEBlock(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, bz):

        fuse_dim = x.size(1)
        x = x.view(bz, -1, fuse_dim)  # [bz, node_num, fuse_dim]
        N = x.size(1)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W, bz)) * v
        x = self.se(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(bz * N, -1)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):

        super(SelfAttention, self).__init__()

        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))

        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):

        input_tensor = x.transpose(1, 0)  # window_size * batch_size * input_size
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w * b * 1
        input_tensor = input_tensor.transpose(1, 0)  # b * w * out
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):

        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class TraceModel(nn.Module):

    def __init__(self, device='cpu', trace_hiddens=[20, 50], trace_kernel_sizes=[3, 3], self_attn=False,
                 chunk_lenth=None,  **kwargs):

        super(TraceModel, self).__init__()

        self.out_dim = trace_hiddens[-1]
        assert len(trace_hiddens) == len(trace_kernel_sizes)

        self.diffusion_model = Diffusion(node_feat_dim=kwargs.get("node_feat_dim"), input_dim=2, output_dim=2, device=device)
        self.diffusion_model.to(device)



        self.net = TransformerEncoder(input_dim=2, d_model=self.out_dim)



        self.self_attn = self_attn
        if self_attn:

            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x: torch.tensor):  #[bz, T, 1]

        x = self.diffusion_model(x)


        hidden_states = self.net(x)


        if self.self_attn:
            return self.attn_layer(hidden_states)

        return hidden_states[:, -1, :]  #[bz, out_dim]



class MetricModel(nn.Module):

    def __init__(self, metric_num, device='cpu', metric_hiddens=[64, 128], metric_kernel_sizes=[3, 3], self_attn=False,
                 chunk_lenth=None,**kwargs):
        super(MetricModel, self).__init__()
        self.metric_num = metric_num
        self.out_dim = metric_hiddens[-1]
        in_dim = metric_num

        assert len(metric_hiddens) == len(metric_kernel_sizes)

        self.net = TransformerEncoder(input_dim=in_dim, d_model=self.out_dim)

        self.self_attn = self_attn
        if self_attn:
            assert (chunk_lenth is not None)
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x):  #[bz, T, metric_num]
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)

        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]  #[bz, out_dim]


class LogModel(nn.Module):

    def __init__(self, event_num, out_dim):
        super(LogModel, self).__init__()
        self.embedder = nn.Linear(event_num, out_dim)

    def forward(self, paras: torch.tensor):  #[bz, event_num]

        return self.embedder(paras)









class MultiSourceEncoder(nn.Module):

    def __init__(self, event_num, metric_num, node_num, device, log_dim=64, fuse_dim=64, alpha=0.5, **kwargs):
        super(MultiSourceEncoder, self).__init__()
        self.node_num = node_num
        self.alpha = alpha


        self.trace_model = TraceModel(device=device, **kwargs)
        trace_dim = self.trace_model.out_dim

        self.log_model = LogModel(event_num, log_dim)


        self.metric_model = MetricModel(metric_num, device=device, **kwargs)
        metric_dim = self.metric_model.out_dim


        fuse_in = trace_dim + log_dim + metric_dim

        if not fuse_dim % 2 == 0:
            fuse_dim += 1

        self.fuse = nn.Linear(fuse_in, fuse_dim)

        self.cglu = CGLU(in_features=fuse_in, hidden_features=fuse_dim, out_features=fuse_dim, act_layer=nn.GELU)
        self.feat_in_dim = int(fuse_dim)


        self.status_model = GraphModel(in_dim=self.feat_in_dim, device=device, **kwargs)
        self.feat_out_dim = self.status_model.out_dim



    def forward(self, graph):

        trace_embedding = self.trace_model(graph.ndata["traces"])  #[bz*node_num,  trace_dim]
        log_embedding = self.log_model(graph.ndata["logs"])  #[bz*node_num, log_dim]
        metric_embedding = self.metric_model(graph.ndata["metrics"])  #[bz*node_num, metric_dim]

        feature = torch.cat((trace_embedding, log_embedding, metric_embedding),dim=-1)  # [bz*node_num, fuse_in]

        bz = feature.size(0) // self.node_num

        feature = self.cglu(feature, H=1, W=1, bz=bz)  # [bz*node_num, fuse_dim]

        embeddings = self.status_model(graph, feature)  #[bz, graph_dim]
        return embeddings

class FullyConnected(nn.Module):


    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):  #[batch_size, in_dim]
        return self.net(x)


import numpy as np


class MainModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, alpha=0.5, debug=False, **kwargs):
        super(MainModel, self).__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha
        self.beta = kwargs.get('beta')


        self.encoder = MultiSourceEncoder(event_num, metric_num, node_num, device, debug=debug, alpha=alpha, **kwargs)

        self.detector = FullyConnected(self.encoder.feat_out_dim, 2, kwargs['detect_hiddens']).to(device)

        self.decoder_criterion = nn.CrossEntropyLoss()

        self.locator = FullyConnected(self.encoder.feat_out_dim, node_num, kwargs['locate_hiddens']).to(device)

        self.locator_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.get_prob = nn.Softmax(dim=-1)


    def forward(self, graph, fault_indexs):

        batch_size = graph.batch_size
        embeddings = self.encoder(graph)  #[bz, feat_out_dim]


        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device)
        for i in range(batch_size):
            if fault_indexs[i] > -1:
                y_prob[i, fault_indexs[i]] = 1


        y_anomaly = torch.zeros(batch_size).long().to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(fault_indexs[i] > -1)

        locate_logits = self.locator(embeddings)
        locate_loss = self.locator_criterion(locate_logits, fault_indexs.to(self.device))

        detect_logits = self.detector(embeddings)
        detect_loss = self.decoder_criterion(detect_logits, y_anomaly)

        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss

        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self.inference(batch_size, node_probs, detect_logits)

        return {'loss': loss, 'y_pred': y_pred, 'y_prob': y_prob.detach().cpu().numpy(), 'pred_prob': node_probs}

    def inference(self, batch_size, node_probs, detect_logits=None):

        node_list = np.flip(node_probs.argsort(axis=1), axis=1)

        y_pred = []

        for i in range(batch_size):

            detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1).squeeze()

            if detect_pred[i] < 1:
                y_pred.append([-1])
            else:
                y_pred.append(node_list[i])

        return y_pred

