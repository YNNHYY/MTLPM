import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib import pyplot as plt

from utils.com_uses import MMSE
from utils.result_display import drawloss
from dataset import trainDataset, valDataset,testDataset
from model import CityModel, GlobalModel
from torch_geometric.nn import MetaLayer

parser = argparse.ArgumentParser(description='Multi-city AQI forecasting')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--run_times', type=int, default=5, help='')#运行5次求平均
parser.add_argument('--epoch', type=int, default=300, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--div_num', type=int, default=13, help='')
parser.add_argument('--gnn_h', type=int, default=32, help='')
parser.add_argument('--rnn_h', type=int, default=64, help='')
parser.add_argument('--rnn_l', type=int, default=1, help='')
parser.add_argument('--aqi_em', type=int, default=16, help='')
parser.add_argument('--poi_em', type=int, default=8, help='poi embedding')
parser.add_argument('--wea_em', type=int, default=12, help='wea embedding')
parser.add_argument('--lr', type=float, default=0.0001, help='lr')
parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
parser.add_argument('--pred_step', type=int, default=72, help='step')
parser.add_argument('--enable-cuda', default=True, help='Enable CUDA')

parser.add_argument('--enc_in', type=int, default=32, help='encoder input size')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)
parser.add_argument('--dec_in', type=int, default=32, help='decoder input size')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
args = parser.parse_args()

device = args.device
Loss_list = []
train_dataset = trainDataset()
train_loader = Data.DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               num_workers=0,
                               shuffle=True)

val_dataset = valDataset()
val_loader = Data.DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=True)
test_dataset = testDataset()
test_loader = Data.DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              num_workers=0,
                              shuffle=False)
for runtimes in range(args.run_times):

    global_model = GlobalModel(args.aqi_em, args.rnn_h, args.rnn_l,
                               args.gnn_h).to(device)
    div1_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div2_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div3_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div4_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div5_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div6_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                      args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                      args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                      args.output_attention, args.d_model, args.n_heads,
                                      args.d_ff, args.activation, args.distil, args.mix).to(device)
    div7_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div8_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                      args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                      args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                      args.output_attention, args.d_model, args.n_heads,
                                      args.d_ff, args.activation, args.distil, args.mix).to(device)
    div9_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div10_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div11_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div12_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)
    div13_model = CityModel(args.aqi_em, args.poi_em, args.wea_em,
                                  args.rnn_h, args.rnn_l, args.gnn_h, args.dec_in ,args.enc_in,args.c_out,
                                  args.e_layers,args.d_layers, args.attn, args.factor, args.dropout,
                                  args.output_attention, args.d_model, args.n_heads,
                                  args.d_ff, args.activation, args.distil, args.mix).to(device)

    global_model = torch.nn.DataParallel(global_model, device_ids=[0])#多个GPU来加速训练
    div1_model = torch.nn.DataParallel(div1_model, device_ids=[0])
    div2_model = torch.nn.DataParallel(div2_model, device_ids=[0])
    div3_model = torch.nn.DataParallel(div3_model, device_ids=[0])
    div4_model = torch.nn.DataParallel(div4_model, device_ids=[0])
    div5_model = torch.nn.DataParallel(div5_model, device_ids=[0])
    div6_model = torch.nn.DataParallel(div6_model, device_ids=[0])
    div7_model = torch.nn.DataParallel(div7_model, device_ids=[0])
    div8_model = torch.nn.DataParallel(div8_model, device_ids=[0])
    div9_model = torch.nn.DataParallel(div9_model, device_ids=[0])
    div10_model = torch.nn.DataParallel(div10_model, device_ids=[0])
    div11_model = torch.nn.DataParallel(div11_model, device_ids=[0])
    div12_model = torch.nn.DataParallel(div12_model, device_ids=[0])
    div13_model = torch.nn.DataParallel(div13_model, device_ids=[0])
    # jiaxing_model = torch.nn.DataParallel(jiaxing_model, device_ids=[0])

    div_model_num = sum(p.numel() for p in global_model.parameters()
                         if p.requires_grad)
    print('city_model:', 'Trainable,', div_model_num)

    div1_model_num = sum(p.numel() for p in div1_model.parameters()
                         if p.requires_grad)
    div2_model_num = sum(p.numel() for p in div2_model.parameters()
                             if p.requires_grad)
    div3_model_num = sum(p.numel() for p in div3_model.parameters()
                         if p.requires_grad)
    div4_model_num = sum(p.numel() for p in div4_model.parameters()
                         if p.requires_grad)
    div5_model_num = sum(p.numel() for p in div5_model.parameters()
                         if p.requires_grad)
    div6_model_num = sum(p.numel() for p in div6_model.parameters()
                         if p.requires_grad)
    div7_model_num = sum(p.numel() for p in div7_model.parameters()
                         if p.requires_grad)
    div8_model_num = sum(p.numel() for p in div8_model.parameters()
                         if p.requires_grad)
    div9_model_num = sum(p.numel() for p in div9_model.parameters()
                         if p.requires_grad)
    div10_model_num = sum(p.numel() for p in div10_model.parameters()
                         if p.requires_grad)
    div11_model_num = sum(p.numel() for p in div11_model.parameters()
                         if p.requires_grad)
    div12_model_num = sum(p.numel() for p in div12_model.parameters()
                         if p.requires_grad)
    div13_model_num = sum(p.numel() for p in div13_model.parameters()
                         if p.requires_grad)
    criterion = nn.MSELoss()
    params = list(global_model.parameters()) + list(div1_model.parameters()) + list(div2_model.parameters()) + \
     list(div3_model.parameters()) + list(div4_model.parameters()) + list(div5_model.parameters()) + \
     list(div6_model.parameters()) + list(div7_model.parameters()) + list(div8_model.parameters()) + \
     list(div9_model.parameters()) + list(div10_model.parameters()) + list(div11_model.parameters()) + \
     list(div12_model.parameters()) + list(div13_model.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    val_loss_min = np.inf
    for epoch in range(args.epoch):
        for i, (divs_data, div1_data,div2_data,div3_data,div4_data,div5_data,div6_data,div7_data,div8_data,
                div9_data,div10_data,div11_data,div12_data,div13_data) in enumerate(train_loader):

            divs_aqi, divs_conn, divs_sim,divs_weather = [x.to(device) for x in divs_data]
            # 计算全局的城市属性[128,24,10,32]，将和站点的气象合并组成最终站点的全局属性
            div_u = global_model(divs_aqi, divs_conn, divs_sim, args.div_num)

            div1_data = [item.to(device, non_blocking=True) for item in div1_data]
            div1_outputs = div1_model(div1_data, div_u[:, :, 0], device)  # div1对应编号为0
            div1_loss = criterion(div1_outputs, div1_data[-1])

            div2_data = [item.to(device, non_blocking=True) for item in div2_data]
            div2_outputs = div2_model(div2_data, div_u[:, :, 1], device)#
            div2_loss = criterion(div2_outputs, div2_data[-1])

            div3_data = [item.to(device, non_blocking=True) for item in div3_data]
            div3_outputs = div3_model(div3_data, div_u[:, :, 2], device)  #
            div3_loss = criterion(div3_outputs, div3_data[-1])

            div4_data = [item.to(device, non_blocking=True) for item in div4_data]
            div4_outputs = div4_model(div4_data, div_u[:, :, 3], device)
            div4_loss = criterion(div4_outputs, div4_data[-1])

            div5_data = [item.to(device, non_blocking=True) for item in div5_data]
            div5_outputs = div5_model(div5_data, div_u[:, :, 4], device)
            div5_loss = criterion(div5_outputs, div5_data[-1])

            div6_data = [item.to(device, non_blocking=True) for item in div6_data]
            div6_outputs = div6_model(div6_data, div_u[:, :, 5], device)
            div6_loss = criterion(div6_outputs, div6_data[-1])

            div7_data = [item.to(device, non_blocking=True) for item in div7_data]
            div7_outputs = div7_model(div7_data, div_u[:, :, 6], device)
            div7_loss = criterion(div7_outputs, div7_data[-1])

            div8_data = [item.to(device, non_blocking=True) for item in div8_data]
            div8_outputs = div8_model(div8_data, div_u[:, :, 7], device)
            div8_loss = criterion(div8_outputs, div8_data[-1])

            div9_data = [item.to(device, non_blocking=True) for item in div9_data]
            div9_outputs = div9_model(div9_data, div_u[:, :, 8], device)
            div9_loss = criterion(div9_outputs, div9_data[-1])
            # div9_loss = MMSE(div9_outputs, div9_data[-1], div9_data[3])
            #
            div10_data = [item.to(device, non_blocking=True) for item in div10_data]
            div10_outputs = div10_model(div10_data, div_u[:, :, 9], device)
            div10_loss = criterion(div10_outputs, div10_data[-1])

            div11_data = [item.to(device, non_blocking=True) for item in div11_data]
            div11_outputs = div11_model(div11_data, div_u[:, :, 10], device)
            div11_loss = criterion(div11_outputs, div11_data[-1])

            div12_data = [item.to(device, non_blocking=True) for item in div12_data]
            div12_outputs = div12_model(div12_data, div_u[:, :, 11], device)
            div12_loss = criterion(div12_outputs, div12_data[-1])

            div13_data = [item.to(device, non_blocking=True) for item in div13_data]
            div13_outputs = div13_model(div13_data, div_u[:, :, 12], device)
            div13_loss = criterion(div13_outputs, div13_data[-1])

            #每一个batch时并不需要与其他batch的梯度混合起来累积计算，
            # 因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.可以用model.zero_grad() or optimizer.zero_grad()
            div1_model.zero_grad()
            div2_model.zero_grad()
            div3_model.zero_grad()
            div4_model.zero_grad()
            div5_model.zero_grad()
            div6_model.zero_grad()
            div7_model.zero_grad()
            div8_model.zero_grad()
            div9_model.zero_grad()
            div10_model.zero_grad()
            div11_model.zero_grad()
            div12_model.zero_grad()
            div13_model.zero_grad()
            global_model.zero_grad()

            loss = div1_loss+div2_loss+div3_loss+div4_loss+div5_loss+div6_loss+\
                   div7_loss+div8_loss+div9_loss+div10_loss+div11_loss+div12_loss+div13_loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0 and epoch % 1 == 0:
                Loss_list.append(div1_loss)
                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div1', epoch, args.epoch, i,
                    int(600 / args.batch_size), div1_loss.item()))
                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div5', epoch, args.epoch, i,
                    int(600 / args.batch_size), div5_loss.item()))
                print('{},Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}'.format(
                    'div9', epoch, args.epoch, i,
                    int(600 / args.batch_size), div1_loss.item()))
            if epoch % 10 == 0 and i %120 == 0 and epoch!=0:
               drawloss(Loss_list)


        val_loss = 0
        with torch.no_grad(): #反向传播时就不会自动求导
            # for j, (divs_val,div1_val,div2_val,div3_val,div4_val,div5_val,div6_val,div7_val,div8_val,\
            #         div9_val,div10_val,div11_val,div12_val, div13_val) in enumerate(val_loader):
            for j, (divs_val,div9_val) in enumerate(val_loader):
                divs_aqi_val, divs_conn_val, divs_sim_val, divs_weather = [x.to(device) for x in divs_val]
                #print(cities_aqi.shape, cities_conn.shape,cities_sim.shape,cities_weather.shape)
                div_u_val = global_model(divs_aqi_val, divs_conn_val, divs_sim_val, args.div_num)

                div1_val = [item.to(device, non_blocking=True) for item in div1_val]
                div1_outputs_val = div1_model(div1_val, div_u_val[:, :, 0], device)
                div1_loss_val = criterion(div1_outputs_val, div1_val[-1])

                div2_val = [item.to(device, non_blocking=True) for item in div2_val]
                div2_outputs_val = div2_model(div2_val, div_u_val[:, :, 1], device)
                div2_loss_val = criterion(div2_outputs_val, div2_val[-1])

                div3_val = [item.to(device, non_blocking=True) for item in div3_val]
                div3_outputs_val = div3_model(div3_val, div_u_val[:, :, 2], device)
                div3_loss_val = criterion(div3_outputs_val, div3_val[-1])

                div4_val = [item.to(device, non_blocking=True) for item in div4_val]
                div4_outputs_val = div4_model(div4_val, div_u_val[:, :, 3], device)
                div4_loss_val = criterion(div4_outputs_val, div4_val[-1])

                div5_val = [item.to(device, non_blocking=True) for item in div5_val]
                div5_outputs_val = div5_model(div5_val, div_u_val[:, :, 4], device)
                div5_loss_val = criterion(div5_outputs_val, div5_val[-1])


                div6_val = [item.to(device, non_blocking=True) for item in div6_val]
                div6_outputs_val = div6_model(div6_val, div_u_val[:, :, 5], device)
                div6_loss_val = criterion(div6_outputs_val, div6_val[-1])
                #
                div7_val = [item.to(device, non_blocking=True) for item in div7_val]
                div7_outputs_val = div7_model(div7_val, div_u_val[:, :, 6], device)
                div7_loss_val = criterion(div7_outputs_val, div7_val[-1])

                div8_val = [item.to(device, non_blocking=True) for item in div8_val]
                div8_outputs_val = div8_model(div8_val, div_u_val[:, :, 7], device)
                div8_loss_val = criterion(div8_outputs_val, div8_val[-1])

                div9_val = [item.to(device, non_blocking=True) for item in div9_val]
                div9_outputs_val = div9_model(div9_val, div_u_val[:, :, 8], device)
                div9_loss_val = criterion(div9_outputs_val, div9_val[-1])
                # div9_loss_val = criterion(div9_outputs_val, div9_val[-1])
                #
                div10_val = [item.to(device, non_blocking=True) for item in div10_val]
                div10_outputs_val = div10_model(div10_val, div_u_val[:, :, 9], device)
                div10_loss_val = criterion(div10_outputs_val, div10_val[-1])

                div11_val = [item.to(device, non_blocking=True) for item in div11_val]
                div11_outputs_val = div11_model(div11_val, div_u_val[:, :, 10], device)
                div11_loss_val = criterion(div11_outputs_val, div11_val[-1])

                div12_val = [item.to(device, non_blocking=True) for item in div12_val]
                div12_outputs_val = div12_model(div12_val, div_u_val[:, :, 11], device)
                div12_loss_val = criterion(div12_outputs_val, div12_val[-1])

                div13_val = [item.to(device, non_blocking=True) for item in div13_val]
                div13_outputs_val = div13_model(div13_val, div_u_val[:, :, 12], device)
                div13_loss_val = criterion(div13_outputs_val, div13_val[-1])

                # jiaxing_val = [item.to(device, non_blocking=True) for item in jiaxing_val]
                # jiaxing_outputs_val = jiaxing_model(jiaxing_val, div_u_val[:, :, 4], device)
                # jiaxing_loss_val = criterion(jiaxing_outputs_val, jiaxing_val[-1])

                val_loss = val_loss+div1_loss_val.item()+div2_loss_val.item()+div3_loss_val.item()+\
                           div4_loss_val.item()+div5_loss_val.item()+div6_loss_val.item()+div7_loss_val.item()+\
                           div8_loss_val.item()+div9_loss_val.item()+div10_loss_val.item()+\
                           div11_loss_val.item()+div12_loss_val.item()+div13_loss_val.item()

                print('val_loss',val_loss)
                if epoch % 20 == 0:
                    print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                        'div1', epoch, args.epoch, j, div9_loss_val.item()))
                    # print('{},Epoch [{}/{}],Step [{}], valLoss: {:.4f}'.format(
                    #     'div5', epoch, args.epoch, j, div5_loss_val.item()))
                    # print('{},Epoch [{}/{}], Step [{}],valLoss: {:.4f}'.format(
                    #     'div9', epoch, args.epoch, j, div9_loss_val.item()))

            if val_loss < val_loss_min and epoch > 50 :
                torch.save(global_model.state_dict(),
                           'global.ckpt')
                torch.save(div1_model.state_dict(),
                           'div1.ckpt')
                torch.save(div2_model.state_dict(),
                           'div2.ckpt')
                torch.save(div3_model.state_dict(),
                           'div3.ckpt')
                torch.save(div4_model.state_dict(),
                           'div4.ckpt')
                torch.save(div5_model.state_dict(),
                           'div5.ckpt')
                torch.save(div6_model.state_dict(),
                           'div6.ckpt')
                torch.save(div7_model.state_dct(),
                           'div7.ckpt')
                torch.save(div8_model.state_dict(),
                           'div8.ckpt')
                torch.save(div9_model.state_dict(),
                           'div9.ckpt')
                torch.save(div10_model.state_dict(),
                           'div10.ckpt')
                torch.save(div11_model.state_dict(),
                           'div11.ckpt')
                torch.save(div12_model.state_dict(),
                           'div12.ckpt')
                torch.save(div13_model.state_dict(),
                           'div13.ckpt')

                val_loss_min = val_loss
                print('save done')

    print('Finished Training')

