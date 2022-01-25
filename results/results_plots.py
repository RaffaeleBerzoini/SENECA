import matplotlib.pyplot as plt
import numpy as np
from results.data import liver_dices, lungs_dices, bones_dices, bladder_dices, kidneys_dices

x = np.arange(5)  # number of grouped barplots

one_T_Watt = [21.97, 20.45, 22.20, 22.06, 23.43]
two_T_Watt = [27.26, 24.27, 27.65, 27.62, 30.56]
four_T_Watt = [28.40, 24.82, 28.54, 28.00, 30.98]
GPU_Watt = [68.01, 67.63, 67.94, 67.56, 67.99]

one_T_FPS = [161.19, 124.27, 137.80, 64.12, 49.53]
two_T_FPS = [303.22, 237.92, 256.01, 123.61, 95.50]
four_T_FPS = [335.40, 254.87, 273.17, 127.91, 98.12]
GPU_FPS = [72.20, 77.45, 65.90, 52.22, 37.23]

width = 0.2

plt.bar(x-0.3, [FPS / WATT for FPS, WATT in zip(GPU_FPS, GPU_Watt)], width, color="#bae4b3")
plt.bar(x-0.1, [FPS / WATT for FPS, WATT in zip(one_T_FPS, one_T_Watt)], width, color='#74c476')
plt.bar(x+0.1, [FPS / WATT for FPS, WATT in zip(two_T_FPS, two_T_Watt)], width, color='#238b45')
plt.bar(x+0.3, [FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)], width, color='#00441b')
plt.xticks(x, ['1', '2', '4', '8', '16'])
plt.xlabel(r'Model Parameters [$\times 10^6$]', fontsize=16)
plt.ylabel(r'Energy Efficiency [$\frac{FPS}{Watt}$]', fontsize=16)
plt.legend(['RTX2060 Mobile', 'ZCU104 1-Thread', 'ZCU104 2-Threads', 'ZCU104 4-Threads'], fontsize=10)
plt.savefig('FPS_VS_Energy.eps', format='eps')
plt.show()

# 4T is always the best model

En_Eff_4_T = [FPS / WATT for FPS, WATT in zip(four_T_FPS, four_T_Watt)]
Dices = [0.9304, 0.9301, 0.9349, 0.9365, 0.9384]

plt.bar(x, [Dice * EEff for Dice, EEff in zip(Dices, En_Eff_4_T)], color='#238b45')
plt.xticks(x, ['1', '2', '4', '8', '16'])
plt.xlabel(r'Model Parameters [$\times 10^6$]', fontsize=16)
plt.ylabel(r'Energy Efficiency $\times$ Dice Score', fontsize=16)
plt.title('Models tested with 4-thread on the Xilinx ZCU104')
plt.savefig('EnEff_x_Acc_VS_params.eps', format='eps')
plt.show()


# Best model = 1M parameters
labels = ['Liver', 'Bladder', 'Lungs', 'Kidneys', 'Bones']
plt.boxplot([np.array(liver_dices), np.array(bladder_dices), np.array(lungs_dices), np.array(kidneys_dices),
             np.array(bones_dices)], vert=True, patch_artist=True, labels=labels)
plt.xticks(fontsize=16)
plt.ylabel('Dice Score', fontsize=16)
plt.savefig('Boxplot_Dices.eps', format='eps')
plt.show()
