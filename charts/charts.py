import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colors = ['#fdd0a2', '#fdae6b', '#41b6c4', '#1d91c0', '#225ea8']
colors = ['#d0d1e6', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d']
chart_fnt = 28
leg_fnt = 24
tick_fnt = 21
tick_num = 8
edge_line_wdth = 0.5
ax_margins = 0.01
width = 0.2
x_tick_num = 5


def plot_runtime(name, dataframe, xlabel, ylabel):

    labels = dataframe.columns.values

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.tick_params(axis='both', which='both', labelsize=tick_fnt)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.tick_params(axis='both', which='both', labelsize=tick_fnt)

    x_ticks = np.arange(x_tick_num)

    # np.array(['1', '2', '4', '8', '16'])
    if len(labels) % 2 == 1:
        offs = np.arange((-len(labels) / 2) * width, (len(labels) / 2) * width + width, width)
    else:
        offs = np.arange((-len(labels) / 2) * width, (len(labels) / 2) * width, width) + 0.1

    i = 0
    for label in labels:
        ax.bar(x_ticks + offs[i], dataframe[label], width=width, color=colors[i], edgecolor='black',
               linewidth=edge_line_wdth, zorder=3)
        i = i + 1

    if len(labels) % 2 == 1:
        ax.set_xticks(x_ticks - width / 2)
    else:
        ax.set_xticks(x_ticks)

    ax.set_xticklabels(list(dataframe.index))
    ax.margins(x=ax_margins)

    ax.set_ylabel(ylabel, fontsize=chart_fnt)
    # ax.set_ylabel(r'Energy Efficiency [$\frac{FPS}{Watt}$]', fontsize=chart_fnt))
    ax.set_xlabel(xlabel, fontsize=chart_fnt)
    ax.grid(visible=True, which='major', linestyle='dotted', axis='y', zorder=0)
    ax.legend(labels, loc='best', ncol=int(len(labels) / 2), fontsize=leg_fnt)

    # removing 0 (zero) from y-axis
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    fig.savefig(name + '.svg', format='svg', bbox_inches='tight', dpi=1200)
    fig.savefig(name + '.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    plt.close(fig)


def runtime_data():
    os.makedirs('JBHI_charts/',exist_ok=True)

    # 2060, Tesla, KV260, Ultra96-V2, ZCU104
    # MobileNetV2 + decoder
    BraTS_Watt = [28.06, 71.58, 12.39, 8.96, 16.47]
    BraTS_FPS = [99.98, 372.52, 96.52, 37.26, 93.34]
    BraTS_EE = [FPS / WATT for FPS, WATT in zip(BraTS_FPS, BraTS_Watt)]

    # vNet
    PROSTATE3T_Watt = [77.22, 81.82, 11.38, 8.59, 15.96]
    PROSTATE3T_FPS = [299.62, 599.48, 145.76, 91.47, 190.68]
    PROSTATE3T_EE = [FPS / WATT for FPS, WATT in zip(PROSTATE3T_FPS, PROSTATE3T_Watt)]

    # U-Net
    CTORG_Watt = [78.17, 80.38, 12.33, 9.59, 17.59]
    CTORG_FPS = [72.09, 199.36, 74.70, 36.52, 82.92]
    CTORG_EE = [FPS / WATT for FPS, WATT in zip(CTORG_FPS, CTORG_Watt)]

    # CNN
    BCCD_Watt = [80.28, 166.56, 12.75, 9.28, 17.35]
    BCCD_FPS = [959.50, 2046.22, 237.92, 114.99, 225.49]
    BCCD_EE = [FPS / WATT for FPS, WATT in zip(BCCD_FPS, BCCD_Watt)]

    dataframe_Watt = pd.DataFrame({'BraTS': BraTS_Watt,
                                   'PROSTATE3T': PROSTATE3T_Watt,
                                   'CT-ORG': CTORG_Watt,
                                   'BCCD': BCCD_Watt})

    dataframe_Watt.index = ['RTX 2060 Mobile', 'Tesla V100', 'Kria KV260', 'Avnet Ultra96-V2', 'ZCU104']

    dataframe_FPS = pd.DataFrame({'BraTS': BraTS_FPS,
                                  'PROSTATE3T': PROSTATE3T_FPS,
                                  'CT-ORG': CTORG_FPS,
                                  'BCCD': BCCD_FPS})

    dataframe_FPS.index = ['RTX 2060 Mobile', 'Tesla V100', 'Kria KV260', 'Avnet Ultra96-V2', 'ZCU104']

    dataframe_EE = pd.DataFrame({'BraTS': BraTS_EE,
                                 'PROSTATE3T': PROSTATE3T_EE,
                                 'CT-ORG': CTORG_EE,
                                 'BCCD': BCCD_EE})

    dataframe_EE.index = ['RTX 2060 Mobile', 'Tesla V100', 'Kria KV260', 'Avnet Ultra96-V2', 'ZCU104']

    global x_tick_num
    x_tick_num = 5
    x_label = 'Device'
    name = "./JBHI_charts/device-watt"
    plot_runtime(name, dataframe_Watt, x_label, 'Watt')
    name = "./JBHI_charts/device-FPS"
    plot_runtime(name, dataframe_FPS, x_label, 'FPS')
    name = "./JBHI_charts/device-EE"
    plot_runtime(name, dataframe_EE, x_label, 'EE')

    global width
    width = 0.15
    x_tick_num = 4
    x_label = 'Dataset'
    name = "./JBHI_charts/dataset-watt"
    plot_runtime(name, dataframe_Watt.T, x_label, 'Watt')
    name = "./JBHI_charts/dataset-FPS"
    plot_runtime(name, dataframe_FPS.T, x_label, 'FPS')
    name = "./JBHI_charts/dataset-EE"
    plot_runtime(name, dataframe_EE.T, x_label, 'EE')


if __name__ == '__main__':
    runtime_data()
