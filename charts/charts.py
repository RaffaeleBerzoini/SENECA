import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from dices_data import *

# colors = ['#fdd0a2', '#fdae6b', '#41b6c4', '#1d91c0', '#225ea8']
colors = ['#d0d1e6','#a6bddb','#74a9cf','#2b8cbe','#045a8d']
chart_fnt=28
leg_fnt=24
tick_fnt=21
tick_num=8
edge_line_wdth=0.5
ax_margins = 0.01
width = 0.2
x_tick_num=5

def plot_boxplots(name):
    dataframe = pd.DataFrame({'Liver':liver_dices, \
    'Bladder':bladder_dices, 'Lungs':lungs_dices, \
    'Kidneys':kidneys_dices, 'Bones':bones_dices })

    labels = dataframe.columns.values
    #Take absolute values of data
    dataframe = dataframe.applymap(lambda x: abs(x))
    for label in labels:
        fig, ax1 = plt.subplots(figsize=(6,6))
        fig.suptitle(label + " distribution")
        ax1.boxplot(dataframe[label])
        plt.scatter([1] * len(dataframe.index), dataframe[label], marker='.')
        ax1.set_xlabel(label)
        ax1.xaxis.set_visible(False)
        ax1.grid(visible=True, which='major', linestyle='dotted')
        image_format = 'svg' # e.g .png, .svg, etc.
        image_name = name + label + '.svg'
        fig.savefig(image_name, format=image_format, bbox_inches = 'tight', dpi=1200)
        plt.close(fig)
    print("Hello moto")

def single_boxplot(name):
    dataframe = pd.DataFrame({'Liver':liver_dices, \
    'Bladder':bladder_dices, 'Lungs':lungs_dices, \
    'Kidneys':kidneys_dices, 'Bones':bones_dices })
    # dataframe = pd.DataFrame({'Liver':liver_dices, \
    # 'Lungs':lungs_dices, \
    # 'Bones':bones_dices })
    labels = dataframe.columns.values
    fig, ax = plt.subplots(1, 1, figsize=(16,10))
    ax.tick_params(axis='both', which='both', labelsize=tick_fnt)

    #for l in labels:
    ax.boxplot(dataframe)
    i=1
    x_labels=np.array(['Liver', 'Bladder','Lungs','Kidneys','Bones'])

    # for l in labels:
    #     plt.scatter([i] * len(dataframe.index), dataframe[l], marker='.')
    #     i=i+1
    ax.set_ylabel(r'Dice Score', fontsize=chart_fnt)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.grid(visible=True, which='major', linestyle='dotted')
    image_format = 'svg' # e.g .png, .svg, etc.
    fig.savefig(name+image_format, format=image_format, bbox_inches = 'tight', dpi=1200)
    fig.savefig(name + '.pdf', format='pdf', bbox_inches = 'tight', dpi=1200)
    plt.close(fig)


def plot_dsc_times_ee(name):
    four_T_Watt = [28.40, 24.82, 28.54, 28.00, 30.98]
    four_T_FPS = [335.40, 254.87, 273.17, 127.91, 98.12]
    En_Eff_4_T = [FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)]
    Dices = [0.9304, 0.9301, 0.9349, 0.9365, 0.9384]
    
    dataframe = pd.DataFrame({'DSC*EE 4 Threads':[dic * ee for dic, ee in zip(Dices, En_Eff_4_T)] })
    maxres = max(dataframe.max())
    # normalized_df = dataframe.apply(lambda x: (x / maxres) )
    normalized_df = dataframe
    labels = dataframe.columns.values


    fig, ax = plt.subplots(1, 1, figsize=(16,10))
    ax.tick_params(axis='both', which='both', labelsize=tick_fnt)
    x_labels=2 ** np.arange(x_tick_num)
    x_ticks=np.arange(x_tick_num)

    i=2
    width = 0.4
    plt.title("Model Tested with 4 threads on the ZCU104", loc='center',fontsize=chart_fnt+2)
    ax.bar(x_ticks, normalized_df[labels[0]], width=width, color=colors[i], edgecolor='black',linewidth=edge_line_wdth, zorder=3)    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    # ax.set_yticks(np.linspace(0,1,num=tick_num))
    # ax.set_yticklabels(map(math.floor, np.linspace(0,maxres,num=tick_num)))

    ax.set_ylabel('Dice Score * Energy Efficiency', fontsize=chart_fnt)
    ax.set_xlabel(r'Model Parameters [$\times 10^6$]', fontsize=chart_fnt)
    ax.grid(visible=True, which='major', linestyle='dotted',axis='y',zorder=0)

    # removing 0 (zero) from y-axis
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    fig.savefig(name + '.svg', format='svg', bbox_inches = 'tight', dpi=1200)
    fig.savefig(name + '.pdf', format='pdf', bbox_inches = 'tight', dpi=1200)
    plt.close(fig)


def plot_runtime(name):
    x = np.arange(5)  # number of grouped barplots

    one_T_Watt = [21.97, 20.45, 22.20, 22.06, 23.43]
    two_T_Watt = [27.26, 24.27, 27.65, 27.62, 30.56]
    four_T_Watt = [28.40, 24.82, 28.54, 28.00, 30.98]
    GPU_Watt = [78.01, 77.63, 77.94, 77.56, 77.99]

    one_T_FPS = [161.19, 124.27, 137.80, 64.12, 49.53]
    two_T_FPS = [303.22, 237.92, 256.01, 123.61, 95.50]
    four_T_FPS = [335.40, 254.87, 273.17, 127.91, 98.12]
    GPU_FPS = [72.20, 77.45, 65.90, 52.22, 37.23]

    # print([FPS / WATT for FPS, WATT in zip(one_T_FPS, one_T_Watt)])
    # print([FPS / WATT for FPS, WATT in zip(two_T_FPS, two_T_Watt)])
    # print([FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)])
    # print([FPS / WATT for FPS, WATT in zip(GPU_FPS, GPU_Watt)])

    dataframe = pd.DataFrame({'ZCU104 1-Thread':[FPS / WATT for FPS, WATT in zip(one_T_FPS, one_T_Watt)], \
        'ZCU104 2-Thread': [FPS / WATT for FPS, WATT in zip(two_T_FPS, two_T_Watt)], \
        'ZCU104 4-Thread': [FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)], \
        'RTX2060 Mobile' : [FPS / WATT for FPS, WATT in zip(GPU_FPS, GPU_Watt)]\
        })
    labels = dataframe.columns.values
    maxres = max(dataframe.max())
    # normalized_df = dataframe.apply(lambda x: (x / maxres) )
    normalized_df = dataframe
    # chart_fnt=28
    # leg_fnt=24
    # tick_fnt=21
    # tick_num=8
    # edge_line_wdth=0.5
    # ax_margins = 0.01
    # width = 0.2
    # x_tick_num=5

    fig, ax = plt.subplots(1, 1, figsize=(16,10))
    ax.tick_params(axis='both', which='both', labelsize=tick_fnt)
    x_labels=2 ** np.arange(x_tick_num)
    x_ticks=np.arange(x_tick_num)

    # np.array(['1', '2', '4', '8', '16'])
    offs = np.arange((-len(labels)/2)*width, (len(labels)/2)*width, width) + 0.1
    i=0
    #for i in range(len(x)):
    for l in labels:
        ax.bar(x_ticks+offs[i], normalized_df[l], width=width, color=colors[i], edgecolor='black',linewidth=edge_line_wdth, zorder=3)
        i=i+1
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.margins(x=ax_margins)
    # ax.set_yticks(np.linspace(0,1,num=tick_num))
    # ax.set_yticklabels(map(math.floor, np.linspace(0,maxres,num=tick_num)))

    ax.set_ylabel('Energy Efficiency [FPS/Watt]', fontsize=chart_fnt)
    # ax.set_ylabel(r'Energy Efficiency [$\frac{FPS}{Watt}$]', fontsize=chart_fnt))
    ax.set_xlabel(r'Model Parameters [$\times 10^6$]', fontsize=chart_fnt)
    ax.grid(visible=True, which='major', linestyle='dotted',axis='y',zorder=0)
    ax.legend(labels, loc='upper right', ncol=int(len(labels)/2),fontsize=leg_fnt)

    # removing 0 (zero) from y-axis
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    # ax.legend(labels, loc='upper left', bbox_to_anchor=(0.0, 1.25), ncol=len(labels),fontsize=leg_fnt)


    # plt.xticks(x, ['1', '2', '4', '8', '16'])
    # plt.xlabel(r'Model Parameters [$\times 10^6$]', fontsize=chart_fnt)
    # plt.bar(x-0.3, [FPS / WATT for FPS, WATT in zip(one_T_FPS, one_T_Watt)], width)
    # plt.bar(x-0.1, [FPS / WATT for FPS, WATT in zip(two_T_FPS, two_T_Watt)], width)
    # plt.bar(x+0.1, [FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)], width)
    # plt.bar(x+0.3, [FPS / WATT for FPS, WATT in zip(GPU_FPS, GPU_Watt)], width)
    # plt.ylabel(r'Energy Efficiency [$\frac{FPS}{Watt}$]', fontsize=14)
    # plt.legend(['ZCU104 1-Thread', 'ZCU104 2-Threads', 'ZCU104 4-Threads', 'RTX2060 Mobile'])
    # plt.show()
    fig.savefig(name + '.svg', format='svg', bbox_inches = 'tight', dpi=1200)
    fig.savefig(name + '.pdf', format='pdf', bbox_inches = 'tight', dpi=1200)
    plt.close(fig)


def main():
    name="./bioai-energyeff"
    plot_runtime(name)
    name="./bioai-dsc-times-ee"
    plot_dsc_times_ee(name)
    name="./bioai-boxplot-"
    plot_boxplots(name)
    name="./bioai-boxplots"
    single_boxplot(name)
if __name__ == '__main__':
    main()