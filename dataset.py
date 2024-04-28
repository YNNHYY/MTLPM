import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import os
import json
import numpy as np
import pandas as pd
import codecs

TIME_WINDOW =72
PRED_TIME = 72
parser = argparse.ArgumentParser(description='Multi-city data processing')
parser.add_argument('--weather_mean',type=float, default=[987.0958, 56.8941, 16.1947,174.3973,1.4650],help='weather mean')
parser.add_argument('--weather_std',type=float, default=[115.6563, 23.5957,10.6629,101.6264,1.5104],help='weather std')
parser.add_argument('--weather_min',type=float, default=[0.0614,0.00230, -18.8, 0.008, 0.0001],help='weather min')
parser.add_argument('--weather_max',type=float, default=[6328.0, 255.0, 50.0,  360.0,  20.0],help='weather max')
args = parser.parse_args()



DATA_PATH = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class trainDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH, 'data generay code', 'stations.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)

        with open(os.path.join(DATA_PATH,'v2_air_data','DIV_train.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                self.divs = json.loads(line)
        print('DIV_sim shape:', np.array(self.divs['sim']).shape)
        print('DIV_weather shape:', np.array(self.divs['weather']).shape)
        print('DIV_pm2.5 shape:', np.array(self.divs['pm2.5']).shape)
        print('DIV_weather length:', len(self.divs['weather']))
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV1_train.txt'), 'r', encoding='utf-8') as f:
            # self.div1 = json.load(f,strict=False)
            for line in f:
                self.div1 = json.loads(line)
        print('div1_sim shape:', np.array(self.div1['sim']).shape)
        print('div1_weather shape:', np.array(self.div1['weather']).shape)
        print('div1_weather_for shape:', np.array(self.div1['weather_for']).shape)
        print('div1_weather length:', len(self.div1['weather']))

        with open(os.path.join(DATA_PATH,'v2_air_data','DIV2_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div2 = json.load(f)
            for line in f:
                self.div2 = json.loads(line)
        print('div2 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV3_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div3 = json.load(f)
            for line in f:
                self.div3 = json.loads(line)
        print('div3 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV4_train.txt'), 'r') as f:
            # self.div4 = json.load(f)
            for line in f:
                self.div4 = json.loads(line)
        print('div4 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV5_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div5 = json.load(f)
            for line in f:
                self.div5 = json.loads(line)
        print('div5 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV6_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div6 = json.load(f)
            for line in f:
                self.div6 = json.loads(line)
        print('div6 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV7_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div7 = json.load(f)
            for line in f:
                self.div7 = json.loads(line)
        print('div7 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV8_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div8 = json.load(f)
            for line in f:
                self.div8 = json.loads(line)
        print('div8 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV9_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div9 = json.load(f)\
            for line in f:
                self.div9 = json.loads(line)
        print('div9 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV10_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div10 = json.load(f)
            for line in f:
                self.div10 = json.loads(line)
        print('div10 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV11_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div11 = json.load(f)
            for line in f:
                self.div11 = json.loads(line)
        print('div11 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV12_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div12 = json.load(f)
            for line in f:
                self.div12 = json.loads(line)
        print('div12 done')
        with open(os.path.join(DATA_PATH,'v2_air_data','DIV13_train.txt'), 'r',
                  encoding='utf-8') as f:
            # self.div13 = json.load(f)
            for line in f:
                self.div13 = json.loads(line)
        print('div13 done')
        self._norm()

    def _norm(self):#归一化
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min)
                                / args.weather_max-args.weather_min).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                   args.weather_max-args.weather_min).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for'])-args.weather_min) /
                                       args.weather_max-args.weather_min).tolist()
        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div4['weather'] = ((np.array(self.div4['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div4['weather_for'] = ((np.array(self.div4['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div7['weather'] = ((np.array(self.div7['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div7['weather_for'] = ((np.array(self.div7['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div11['weather'] = ((np.array(self.div11['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div11['weather_for'] = ((np.array(self.div11['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div12['weather'] = ((np.array(self.div12['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div12['weather_for'] = ((np.array(self.div12['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        print('norm done')
    def GetDivData(self, div_name, div_source, index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        # div_sim_zeros = torch.zeros_like(div_sim)
        div_conn = torch.tensor(div_source['conn'])
        # div_conn_zeros = torch.zeros_like(div_conn)
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_weather_zeros = torch.zeros_like(div_weather)
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_for_zeros = torch.zeros_like(div_for)
        div_poi = torch.FloatTensor(div_source['poi'])
        # div_poi_zeros = torch.zeros_like(div_poi)

        div_data = [div_con, div_conn, div_poi, div_sim,
                div_weather, div_for, div_y]
        # print('getdivdata done')
        return div_data

    def __getitem__(self, index):
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        div11_data = self.GetDivData('DIV11', self.div11, index)
        div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        # divs_conn_zeros = torch.zeros_like(divs_conn)
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_weather_zeros = torch.zeros_like(divs_weather)
        divs_sim = torch.FloatTensor(self.divs['sim'][index])
        # divs_sim_zeros = torch.zeros_like(divs_sim)

        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data,div1_data,div2_data,div3_data,div4_data,div5_data,div6_data,\
               div7_data,div8_data,div9_data,div10_data,div11_data,div12_data,div13_data


    def __len__(self):
        return len(self.div1['weather'])


class valDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH, 'data generay code', 'stations.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)

        with open(os.path.join(DATA_PATH,'v2_air_data','DIV_val.txt'), 'r',encoding='utf-8') as f:
            # self.divs = json.load(f)
            for line in f:
                self.divs= json.loads(line)
        print("the val numbers of div:", len(self.divs['pm2.5']))
        print("the val div_weather shape:", (np.array(self.divs['weather'])).shape)
        print("the val div_sim shape:", (np.array(self.divs['sim'])).shape)

        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV1_val.txt'), 'r',encoding='utf-8') as f:
            # self.div1 = json.load(f)
            for line in f:
                self.div1 = json.loads(line)
        print("the val numbers of div1:", len(self.div1['sim']))
        print("the val div1 weather shape:", (np.array(self.div1['weather_for'])).shape)

        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV2_val.txt'), 'r',encoding='utf-8') as f:
            # self.div2 = json.load(f)
            for line in f:
                self.div2 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV3_val.txt'), 'r',encoding='utf-8') as f:
            # self.div3 = json.load(f)
            for line in f:
                self.div3 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV4_val.txt'), 'r',encoding='utf-8') as f:
            # self.div4 = json.load(f)
            for line in f:
                self.div4 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV5_val.txt'), 'r',encoding='utf-8') as f:
            # self.div5 = json.load(f)
            for line in f:
                self.div5 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV6_val.txt'), 'r',encoding='utf-8') as f:
            # self.div6 = json.load(f)
            for line in f:
                self.div6 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV7_val.txt'), 'r',encoding='utf-8') as f:
            # self.div7 = json.load(f)
            for line in f:
                self.div7 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV8_val.txt'), 'r',encoding='utf-8') as f:
            # self.div8 = json.load(f)
            for line in f:
                self.div8 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV9_val.txt'), 'r',encoding='utf-8') as f:
            # self.div9 = json.load(f)
            for line in f:
                self.div9 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV10_val.txt'), 'r',encoding='utf-8') as f:
            # self.div10 = json.load(f)
            for line in f:
                self.div10 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV11_val.txt'), 'r',encoding='utf-8') as f:
            # self.div11 = json.load(f)
            for line in f:
                self.div11 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV12_val.txt'), 'r',encoding='utf-8') as f:
            # self.div12 = json.load(f)
            for line in f:
                self.div12 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV13_val.txt'), 'r',encoding='utf-8') as f:
            # self.div13 = json.load(f)
            for line in f:
                self.div13 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'jiaxing72_val.txt'), 'r',encoding='utf-8') as f:
            # self.div8 = json.load(f)
            for line in f:
                self.jiaxing = json.loads(line)
        self._norm()

    def _norm(self):
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min)
                                / args.weather_max - args.weather_min).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div4['weather'] = ((np.array(self.div4['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div4['weather_for'] = ((np.array(self.div4['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div7['weather'] = ((np.array(self.div7['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div7['weather_for'] = ((np.array(self.div7['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div11['weather'] = ((np.array(self.div11['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div11['weather_for'] = ((np.array(self.div11['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div12['weather'] = ((np.array(self.div12['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div12['weather_for'] = ((np.array(self.div12['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.jiaxing['weather'] = ((np.array(self.jiaxing['weather']) - args.weather_min) /
                                   args.weather_max - args.weather_min).tolist()
        self.jiaxing['weather_for'] = ((np.array(self.jiaxing['weather_for']) - args.weather_for_min) /
                                       args.weather_for_max - args.weather_for_min).tolist()
    #
    def GetDivData(self,div_name,div_source,index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        # div_sim_zeros = torch.zeros_like(div_sim)
        div_conn = torch.tensor(div_source['conn'])
        # div_conn_zeros = torch.zeros_like(div_conn)
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_weather_zeros = torch.zeros_like(div_weather)
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_for_zeros = torch.zeros_like(div_for)
        div_poi = torch.FloatTensor(div_source['poi'])
        # div_poi_zeros = torch.zeros_like(div_poi)

        div_data = [div_con, div_conn, div_poi, div_sim, div_weather, div_for, div_y]

        return div_data

    def __getitem__(self, index):
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        div11_data = self.GetDivData('DIV11', self.div11, index)
        div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        # divs_conn_zeros = torch.zeros_like(divs_conn)
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_weather_zeros = torch.zeros_like(divs_weather)
        divs_sim = torch.FloatTensor(self.divs['sim'][index])
        # divs_sim_zeros = torch.zeros_like(divs_sim)
        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data,div1_data,div2_data,div3_data,div4_data,div5_data,div6_data,\
               div7_data,div8_data,div9_data,div10_data,div11_data, div12_data, div13_data
    def __len__(self):
        return len(self.div1['weather'])

class testDataset(Data.Dataset):
    def __init__(self, transform=None, train=True):
        with open(os.path.join(DATA_PATH, 'data generay code', 'stations.json'), 'r',
                  encoding='utf_8') as f:
            self.stations = json.load(f)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV_test.txt'), 'r',encoding='utf_8') as f:
            # self.divs = json.load(f)
            for line in f:
                self.divs = json.loads(line)
        print("the test.py numbers of div:", len(self.divs['pm2.5']))
        print("the test.py div_weather shape:", (np.array(self.divs['weather'])).shape)
        print("the test.py div_sim shape:", (np.array(self.divs['sim'])).shape)

        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV1_test.txt'), 'r',encoding='utf_8') as f:
            # self.div1 = json.load(f)
            for line in f:
                self.div1 = json.loads(line)
        print("the test.py numbers of div1:", len(self.div1['sim']))
        print("the test.py div1 shape:", (np.array(self.div1['weather_for'])).shape)

        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV2_test.txt'), 'r',encoding='utf_8') as f:
            # self.div2 = json.load(f)
            for line in f:
                self.div2 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV3_test.txt'), 'r',encoding='utf_8') as f:
            # self.div3 = json.load(f)
            for line in f:
                self.div3 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV4_test.txt'), 'r') as f:
            # self.div4 = json.load(f)
            for line in f:
                self.div4 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV5_test.txt'), 'r',encoding='utf_8') as f:
            # self.div5 = json.load(f)
            for line in f:
                self.div5 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV6_test.txt'), 'r',encoding='utf_8') as f:
            # self.div6 = json.load(f)
            for line in f:
                self.div6 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV7_test.txt'), 'r',encoding='utf_8') as f:
            # self.div7 = json.load(f)
            for line in f:
                self.div7 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV8_test.txt'), 'r',encoding='utf_8') as f:
            # self.div8 = json.load(f)
            for line in f:
                self.div8 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV9_test.txt'), 'r',encoding='utf_8') as f:
            # self.div9 = json.load(f)
            for line in f:
                self.div9 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV10_test.txt'), 'r',encoding='utf_8') as f:
            # self.div10 = json.load(f)
            for line in f:
                self.div10 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV11_test.txt'), 'r',encoding='utf_8') as f:
            # self.div11 = json.load(f)
            for line in f:
                self.div11 = json.loads(line)
        with open(os.path.join(DATA_PATH,'v2_air_data', 'DIV12_test.txt'), 'r',encoding='utf_8') as f:
            self.div12 = json.load(f)
            for line in f:
                self.div12 = json.loads(line)
        with open(os.path.join(DATA_PATH, 'v2_air_data', 'DIV13_test.txt'), 'r',encoding='utf_8') as f:
            # self.div13 = json.load(f)
            for line in f:
                self.div13 = json.loads(line)
        self._norm()

    def _norm(self):
        self.divs['weather'] = ((np.array(self.divs['weather']) - args.weather_min)
                                / args.weather_max - args.weather_min).tolist()

        self.div1['weather'] = ((np.array(self.div1['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div1['weather_for'] = ((np.array(self.div1['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div2['weather'] = ((np.array(self.div2['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div2['weather_for'] = ((np.array(self.div2['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div3['weather'] = ((np.array(self.div3['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div3['weather_for'] = ((np.array(self.div3['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div4['weather'] = ((np.array(self.div4['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div4['weather_for'] = ((np.array(self.div4['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div5['weather'] = ((np.array(self.div5['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div5['weather_for'] = ((np.array(self.div5['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div6['weather'] = ((np.array(self.div6['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div6['weather_for'] = ((np.array(self.div6['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div7['weather'] = ((np.array(self.div7['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div7['weather_for'] = ((np.array(self.div7['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div8['weather'] = ((np.array(self.div8['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div8['weather_for'] = ((np.array(self.div8['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div9['weather'] = ((np.array(self.div9['weather']) - args.weather_min) /
                                args.weather_max - args.weather_min).tolist()
        self.div9['weather_for'] = ((np.array(self.div9['weather_for']) - args.weather_min) /
                                    args.weather_max - args.weather_min).tolist()
        self.div10['weather'] = ((np.array(self.div10['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div10['weather_for'] = ((np.array(self.div10['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div11['weather'] = ((np.array(self.div11['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div11['weather_for'] = ((np.array(self.div11['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div12['weather'] = ((np.array(self.div12['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div12['weather_for'] = ((np.array(self.div12['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()
        self.div13['weather'] = ((np.array(self.div13['weather']) - args.weather_min) /
                                 args.weather_max - args.weather_min).tolist()
        self.div13['weather_for'] = ((np.array(self.div13['weather_for']) - args.weather_min) /
                                     args.weather_max - args.weather_min).tolist()


    def GetDivData(self,div_name,div_source,index):
        station_list = self.stations[div_name]
        div_con = []
        div_y = []
        for x in station_list:
            div_con.append(div_source[x][index][:TIME_WINDOW])
            div_y.append(div_source[x][index][TIME_WINDOW:])

        div_con = torch.FloatTensor(div_con)
        div_y = torch.FloatTensor(div_y)
        div_sim = torch.FloatTensor(div_source['sim'][index])
        # div_sim_zeros = torch.zeros_like(div_sim)
        div_conn = torch.tensor(div_source['conn'])
        # div_conn_zeros = torch.zeros_like(div_conn)
        div_weather = torch.FloatTensor(div_source['weather'][index])
        div_weather_zeros = torch.zeros_like(div_weather)
        div_for = torch.FloatTensor(div_source['weather_for'][index])
        div_for_zeros = torch.zeros_like(div_for)
        div_poi = torch.FloatTensor(div_source['poi'])
        # div_poi_zeros = torch.zeros_like(div_poi)

        div_data = [div_con, div_conn, div_poi, div_sim,div_weather, div_for, div_y]

        return div_data

    def __getitem__(self, index): #__getitem__的调用要通过： 对象[index]调用
        div1_data = self.GetDivData('DIV1', self.div1, index)
        div2_data = self.GetDivData('DIV2', self.div2, index)
        div3_data = self.GetDivData('DIV3', self.div3, index)
        div4_data = self.GetDivData('DIV4', self.div4, index)
        div5_data = self.GetDivData('DIV5', self.div5, index)
        div6_data = self.GetDivData('DIV6', self.div6, index)
        div7_data = self.GetDivData('DIV7', self.div7, index)
        div8_data = self.GetDivData('DIV8', self.div8, index)
        div9_data = self.GetDivData('DIV9', self.div9, index)
        div10_data = self.GetDivData('DIV10', self.div10, index)
        div11_data = self.GetDivData('DIV11', self.div11, index)
        div12_data = self.GetDivData('DIV12', self.div12, index)
        div13_data = self.GetDivData('DIV13', self.div13, index)

        divs_con = torch.FloatTensor(self.divs['pm2.5'][index])
        divs_conn = torch.tensor(self.divs['conn'])
        # divs_conn_zeros = torch.zeros_like(divs_conn)
        divs_weather = torch.FloatTensor(self.divs['weather'][index])
        divs_weather_zeros = torch.zeros_like(divs_weather)
        divs_sim = torch.FloatTensor(self.divs['sim'][index])
        # divs_sim_zeros = torch.zeros_like(divs_sim)
        divs_data = [divs_con, divs_conn, divs_sim, divs_weather]

        return divs_data, div1_data, div2_data, div3_data, div4_data, div5_data, div6_data, \
               div7_data, div8_data, div9_data, div10_data, div11_data, div12_data, div13_data
    def __len__(self):
        return len(self.div1['weather'])

