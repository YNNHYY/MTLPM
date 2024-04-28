import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
def draw_site_data(site_id, data_pre, data_ture, air_factor_name):

    y_pred_station = data_pre[site_id-1][0:4800]
    y_true_station = data_ture[site_id-1][0:4800]
    time_list = list(range(len(y_pred_station)))
    data = pd.DataFrame({'pred':y_pred_station,'true':y_true_station})
    data.to_csv('data{}.csv'.format(site_id),index=False)
    plt.figure(figsize=(100, 6))
    plt.grid(axis="y")
    y_major_locator = plt.MultipleLocator(20)
    xticks = range(0,len(time_list)+1,6)
    ax = plt.gca()
    plt.xticks(rotation=300)
    ax.set_xticks(xticks)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(time_list, y_true_station, linewidth=1, linestyle="-", label="true value", c='red')
    plt.plot(time_list, y_pred_station, linewidth=1, linestyle="--", label="predict value", c='green')
    plt.legend(loc='best')  # 显示图例，前提是plot参数里写上label;loc是图例的位置
    plt.ylabel(air_factor_name)
    plt.xlabel('time/h')
    plt.title(site_id, fontsize=16)
    plt.grid(linestyle='-.')
    minx = 0
    maxx = len(time_list)
    miny = min(y_true_station)
    maxy = max(y_true_station)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    save_path = '%s.png' % (site_id)
    plt.savefig(save_path,bbox_inches='tight', pad_inches=0.0)

def drawloss(loss):
    # 将张量从 GPU 移动到 CPU
    loss =  [tensor.cpu().detach().numpy() for tensor in loss]
    time_list = list(range(len(loss)))
    data = pd.DataFrame({'Loss': loss})
    data.to_csv('MSE_loss.csv'.format(data), index=False)

    num_epochs = len(loss)//12
    epoch_positions = np.linspace(0, len(loss) - 1, num_epochs + 1, dtype=int)
    plt.xticks(rotation=300)
    epoch_labels = np.arange(0, num_epochs + 1, 10)
    plt.xticks(epoch_positions[::10], epoch_labels)

    plt.plot(time_list, loss, linewidth=1, linestyle="-", c='red')
    plt.ylabel("MSE")
    plt.xlabel('Epochs')
    minx = 0
    maxx = len(time_list)
    miny = min(loss)
    maxy = max(loss)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    save_path = 'MSE.png'
    print('fig save')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()