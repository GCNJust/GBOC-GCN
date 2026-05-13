# -*- coding: utf-8 -*-
# @Time : 2023/9/22 16:07
# @Author : Teng Qing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.figure(dpi=80)
# plt.style.use('ggplot')
x_axis_data = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

ALOI_data_GCN = [0.9339,0.9432,0.9380,0.9484,0.9081,0.9442,0.9360,0.9401,0.9665,0.9514,0.9399]
ALOI_data_MLAN =[0.9662,0.9616,0.9561,0.9561,0.9644,0.9616,0.9583,0.9688,0.965,0.9627,0.9475]
ALOI_data_LGCN_FF = [0.7813,0.8148, 0.8253,0.8154,0.7989,0.8083,0.8571,0.8553,0.8619,0.9,0.8213]
ALOI_data_JFGCN = [0.9555,0.9645,0.9638,0.9576,0.9576,0.9451,0.9597,0.9597,0.9708,0.9513,0.9504]
ALOI_data_HGCN_MVSC= [0.9594,0.9465,0.9513,0.9594,0.9432,0.9497,0.9546,0.9513,0.9622,0.9497,0.9216]
ALOI_data_Co_GCN =[0.9511,0.9368,0.9511,0.9531,0.9450,0.9511,0.9511,0.9511,0.9511,0.9511,0.9411]
ALOI_data_DSIGCN = [0.9437,0.9619,0.9528,0.9545,0.9623,0.9483,0.9565,0.9599,0.9574,0.9334,0.9101]
ALOI_data_MAGCN = [0.6912,0.6723,0.7012,0.7213,0.7112,0.73810,0.7023,0.7215,0.7033,0.7165,0.7023]
ALOI_data_KNN = [0.9512,0.9623,0.9725,0.9663,0.9764,0.9698,0.9828,0.9712,0.9723,0.9436,0.9615]
name = "fig20"

plt.plot(x_axis_data, ALOI_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='ALOI')
plt.plot(x_axis_data, ALOI_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MNIST')
# plt.plot(x_axis_data, ALOI_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='MSRC-v1')
plt.plot(x_axis_data, ALOI_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='NUS-WIDE')
plt.plot(x_axis_data, ALOI_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='BBCnews')
# plt.plot(x_axis_data, ALOI_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='BBCsports')
plt.plot(x_axis_data, ALOI_data_DSIGCN, 'rs-', alpha=0.5, linewidth=2, label='OutScene')
plt.plot(x_axis_data, ALOI_data_MAGCN, 'v-', color='purple', alpha=0.5, linewidth=2, label='ORL')

plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 " )  # 显示上面的label
plt.xlabel('Parameter μ',fontsize = '14')  # x_label
plt.yticks(np.arange(0.4, 1.05, 0.1))
plt.xticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='lower right', markerscale=1.5)
plt.ylabel('ACC',fontsize = '14')  # y_label
plt.grid(True, linestyle="--", alpha=0.5)
y = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.savefig('./result/mu/'+name+'.png', dpi=1000)
plt.show()