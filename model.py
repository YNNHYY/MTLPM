import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
from embed import DataEmbedding
from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from attn import FullAttention, ProbAttention, AttentionLayer
from decoder import Decoder, DecoderLayer

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True)

    def forward(self, x, h0,c0):

        out, (h_n,c_n) = self.lstm(x, h0,c0)

        return h_n,c_n


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, h_0,c_0):
        out, (h_n, c_n) = self.lstm(x, (h_0,c_0))
        out = self.lin(out)
        out = self.relu(out)
        return out, h_n,c_n


class GlobalModel(torch.nn.Module):
    def __init__(self, aqi_em, rnn_h, rnn_l, gnn_h):
        super(GlobalModel, self).__init__()
        self.aqi_em = aqi_em
        self.rnn_h = rnn_h
        self.gnn_h = gnn_h
        self.aqi_embed = Seq(Lin(1, aqi_em), ReLU())
        self.aqi_rnn = nn.LSTM(aqi_em, rnn_h, rnn_l,batch_first=True)
        self.city_gnn = CityGNN(rnn_h, 2, gnn_h)

    def batchInput(self, x, edge_w, edge_conn):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_conn.size(0)):
            edge_conn[i, :] = torch.add(edge_conn[i, :], i * sta_num)
        edge_conn = edge_conn.transpose(0, 1)
        edge_conn = edge_conn.reshape(2, -1)
        return x, edge_w, edge_conn

    def forward(self, city_aqi, city_conn, city_w, city_num):
        city_aqi = city_aqi.unsqueeze(dim=-1)
        city_aqi = self.aqi_embed(city_aqi)
        city_aqi, _ = self.aqi_rnn(city_aqi.reshape(-1, 72, self.aqi_em))
        city_aqi = city_aqi.reshape(-1, city_num, 72, self.rnn_h)
        city_aqi = city_aqi.transpose(1, 2)
        city_aqi = city_aqi.reshape(-1, city_num, city_aqi.shape[-1])

        city_conn = city_conn.transpose(1, 2).repeat(72, 1, 1)
        city_w = city_w.reshape(-1, city_w.shape[-2], city_w.shape[-1])
        # print(city_aqi.shape,city_conn.shape, city_w.shape)
        city_x, city_weight, city_conn = self.batchInput(
            city_aqi, city_w, city_conn)
        out = self.city_gnn(city_x, city_conn, city_weight)  # 城市间更新
        out = out.reshape(-1, 72, city_num, out.shape[-1])

        return out  # city_u


class CityGNN(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h):
        super(CityGNN, self).__init__()
        self.node_mlp_1 = Seq(Lin(2 * node_h + edge_h, gnn_h),
                              ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h, gnn_h), ReLU(inplace=True))
        # 层内传递 城市

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))  # 平均值 即站点到城市
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class CityModel(nn.Module):
    """Station graph"""

    def __init__(self, aqi_em, poi_em, wea_em, rnn_h, rnn_l, gnn_h, dec_in,enc_in,c_out,
                 e_layers=3, d_layers=2, attn='prob', factor=5, dropout=0.0, output_attention=False,
                 d_model=512, n_heads=8, d_ff=512, activation='gelu', distil=True, mix=True):
        super(CityModel, self).__init__()
        self.rnn_h = rnn_h
        self.gnn_h = gnn_h
        self.rnn_l = rnn_l

        # 嵌入
        self.aqi_embed = Seq(Lin(1, aqi_em), ReLU())  # 乘权重参数后经过relu激活函数
        self.poi_embed = Seq(Lin(5, poi_em), ReLU())
        self.city_embed = Seq(Lin(gnn_h, wea_em), ReLU())
        self.wea_embed = Seq(Lin(5, wea_em), ReLU())
        # self.for_embed = Seq(Lin(4, wea_em), ReLU())
        self.out = nn.Linear(32, 1, bias=True)
        self.putin = Seq(Lin(1,32),ReLU())
        self.sta_gnn = StaGNN(aqi_em + poi_em, 2, gnn_h, 2 * wea_em)
        self.encoder = RNNEncoder(input_size=gnn_h,
                                  hidden_size=rnn_h,
                                  num_layers=rnn_l)
        self.decoder_embed = Seq(Lin(1, aqi_em), ReLU())
        self.decoder = RNNDecoder(input_size=5 + aqi_em,
                                  hidden_size=rnn_h,
                                  num_layers=rnn_l)

        self.embed = DataEmbedding(enc_in)
        self.embed2 = DataEmbedding(dec_in)

        self.attn = attn
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encode = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decode = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
    def batchInput(self, x, edge_w, edge_conn):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_conn.size(0)):
            edge_conn[i, :] = torch.add(edge_conn[i, :], i * sta_num)
        edge_conn = edge_conn.transpose(0, 1)
        edge_conn = edge_conn.reshape(2, -1)
        return x, edge_w, edge_conn

    def forward(self, city_data, city_u, device):
        sta_aqi, sta_conn, sta_poi, sta_w, sta_wea, sta_for, sta_y = city_data #stay 32 2 12
        sta_num = sta_aqi.shape[1]  # 站点数
        sta_x = sta_aqi.unsqueeze(dim=-1)  # 在-1处升一维 32 2 24 ->32 2 24 1
        sta_x = self.aqi_embed(sta_x)  # 32 2 24 16
        sta_poi = self.poi_embed(sta_poi)  # 32 2 8 原始 32 2 5
        sta_poi = sta_poi.unsqueeze(dim=-2).repeat_interleave(72, dim=-2)  # 32 2 24 8
        sta_x = torch.cat([sta_x, sta_poi], dim=-1)  # 32 2 24 24 空气质量+位置信息
        sta_x = sta_x.transpose(1, 2)  # 32 24 2 24
        sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1])  # 768 2 24

        sta_conn = sta_conn.transpose(1, 2).repeat(72, 1, 1)  # 768 2 2
        sta_w = sta_w.reshape(-1, sta_w.shape[-2], sta_w.shape[-1])  # sim 32 24 2 ->768 2 2
        sta_x, sta_weight, sta_conn = self.batchInput(sta_x, sta_w, sta_conn)  # 1536 24   1536 2   2 1536

        city_u = self.city_embed(city_u)  # city_u 向下更新中城市的向下更新值，与站点天气数据结合  32 24 12
        sta_wea = self.wea_embed(sta_wea)  # 32 24 12
        sta_u = torch.cat([city_u, sta_wea], dim=-1)  # 32 24 24

        sta_x = self.sta_gnn(sta_x, sta_conn, sta_weight, sta_u, sta_num)  # 站内更新 1536 32
        sta_x = sta_x.reshape(-1,72, sta_num, sta_x.shape[-1]).transpose(1, 2)  # 32 2 24 32
        sta_x = sta_x.reshape(-1,72, sta_x.shape[-1])  # 64 72 32 时间数据)

        data_in = sta_x#64 24 32
        data_he = sta_x[:,36:72,:]# 64 12 32
        data_out = torch.zeros([sta_x.shape[0],72, sta_x.shape[-1]]).float().to(device)#64 72 32
        data_out = torch.cat([data_he,data_out ], dim=-2)#64 108 32
        data_in = self.embed(data_in) # 64 72 512

        enc_out, attns = self.encode(data_in,attn_mask=None)# 64 72 512

        dec_out = self.embed2(data_out) #32 2 512   64 108 512
        dec_out = self.decode(dec_out, enc_out, x_mask=None, cross_mask=None)# 32 2 512   64 24 512'
        dec_out = self.projection(dec_out) # 32 2 512->32


        dec_out = dec_out.reshape(sta_aqi.shape[0],sta_num,-1) # 32 2 24

        return dec_out[:,:,-72:] # 输出预测结果


class StaGNN(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h, u_h):
        super(StaGNN, self).__init__()
        self.node_mlp_1 = Seq(Lin(2 * node_h + edge_h, gnn_h),
                              ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h + u_h, gnn_h),
                              ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr, u, sta_num):
        u = u.reshape(-1, u.shape[-1])
        u = u.repeat(sta_num, 1)
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u], dim=1)
        return self.node_mlp_2(out)
