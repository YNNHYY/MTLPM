import  math
import  numpy as np
import torch
from utils.math_utils import mse

def get_sim_wind_array(wind_directions,start_station_localtion,end_station_localtion):
    ab =((start_station_localtion[0]-end_station_localtion[0])+
         (start_station_localtion[1]-start_station_localtion[1]))
    a = len(wind_directions)
    b= wind_directions[1][0]
    ws = [wind_directions[i][0] * ab / abs(wind_directions[i][0]) / abs(ab) for i in range(a)]



    return ws
def get_sim_location(start_station_localtion,end_station_localtion):
    ab = math.sqrt((start_station_localtion[0]-end_station_localtion[0])**2+
                   (start_station_localtion[1]-start_station_localtion[1])**2)
    gs = 1/ab
    return  gs

def MMSE(pred,true,weight):
    ns = 0
    ks = 0
    nn = 0
    kn = 0
    mmse= 0
    for i in range(pred.shape[1]):
        for j in range(pred.shape[1]):
            if i != j:
                local = i * pred.shape[1] + j - 1
                if local > 5 and local <= 10:
                    local = local - 1
                if local > 10:
                    local = local - 2
                gs = weight[:, :, local, :][0, 0, 0]
                ws = weight[:, :, local, :][0, 0, 1]
                if gs*ws > 0:
                    n = true[:,j,:]
                    test = n/3
                    ns = ns + n
                    nn = nn + 1
                if gs*ws < 0:
                    k = torch.mean(true[:,j,:]).item()
                    ks = ks + k
                    kn = kn + 1
        if nn and kn != 0:
            e1 = ns/nn
            e2 = ks/kn
            loss1 = mse(true[:,i,:],e1)
            loss2 = mse(true[:,i,:],e2)

        if nn == 0:
            e2 = ks / kn
            loss1 = 1
            loss2 = mse(true[:, i, :], e2)
        if kn == 0:
            e1 = ns / nn
            loss1 = mse(true[:,i,:],e1)
            loss2 = 1
        loss = mse(pred[:, i, :], true[:, i, :]) + loss1 / loss2
        mmse = mmse + loss
    mmse = mmse/weight.shape[2]
    return  mmse



