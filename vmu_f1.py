# -*- coding: utf-8 -*-
# @Time : 2023/9/22 16:07
# @Author : Teng Qing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.figure(dpi=80)
# plt.style.use('ggplot')
x_axis_data = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

ALOI_data_GCN = [0.9014,0.9421,0.9368,0.9351,0.9281,0.9442,0.9360,0.9391,0.9665,0.9422,0.9319]
ALOI_data_MLAN =[0.9399,0.9492,0.9467,0.9508,0.9461,0.9625,0.96,0.9655,0.965,0.9657,0.9455]
ALOI_data_LGCN_FF = [0.7453,0.8108, 0.8213,0.7724,0.8019,0.7883,0.8641,0.8293,0.8129,0.767,0.7313]
ALOI_data_JFGCN = [0.9195,0.9495,0.9468,0.9506,0.9466,0.9621,0.9507,0.9647,0.9508,0.9513,0.9494]
ALOI_data_HGCN_MVSC= [0.9184,0.9325,0.9383,0.9484,0.9282,0.9377,0.9436,0.9693,0.9232,0.9267,0.9346]
ALOI_data_Co_GCN =[0.9341,0.92068,0.9341,0.9341,0.9280,0.9341,0.9341,0.93411,0.9341,0.9341,0.9321]
ALOI_data_DSIGCN = [0.9227,0.9529,0.9528,0.9615,0.9483,0.9563,0.9595,0.9569,0.9314,0.9314,0.9411]
ALOI_data_MAGCN = [0.7112,0.6423,0.7012,0.7013,0.6912,0.6981,0.7023,0.6915,0.6833,0.7236,0.6523]
ALOI_data_KNN = [0.8812,0.8923,0.8925,0.9143,0.9064,0.8998,0.8828,0.854,0.9023,0.8936,0.8865]

name = "fig21"

plt.plot(x_axis_data, ALOI_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='ALOI')
plt.plot(x_axis_data, ALOI_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MNIST')
# plt.plot(x_axis_data, ALOI_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='MSRC-v1')
plt.plot(x_axis_data, ALOI_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='NUS-WIDE')
plt.plot(x_axis_data, ALOI_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='BBCnews')
# plt.plot(x_axis_data, ALOI_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='BBCsports')
plt.plot(x_axis_data, ALOI_data_DSIGCN, 'rs-', alpha=0.5, linewidth=3, label='OutScene')
plt.plot(x_axis_data, ALOI_data_MAGCN, 'v-', color='purple', alpha=0.5, linewidth=2, label='ORL')

plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 " )  # 显示上面的label
plt.xlabel('Parameter μ',fontsize = '14')  # x_label
plt.yticks(np.arange(0.4, 1.05, 0.1))
plt.xticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='lower right', markerscale=1.5)
plt.ylabel('F1',fontsize = '14')  # y_label
plt.grid(True, linestyle="--", alpha=0.5)
y = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# plt.yticks(y)
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig('./result/mu/'+name+'.png', dpi=1000)
plt.show()