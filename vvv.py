# -*- coding: utf-8 -*-
# @Time : 2023/9/22 16:07
# @Author : Teng Qing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

plt.figure(dpi=80)
# plt.style.use('ggplot')
x_axis_data = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]


ALOI_data_GCN = [0.6527,0.8562,0.9043,0.9124,0.9237,0.9256,0.9278,0.9313,0.9351,0.9366]
ALOI_data_MLAN =[0.7223,0.8918,0.8516,0.8894,0.8913,0.9012,0.9121,0.9143,0.9155,0.9263]
ALOI_data_DSRL =[0.8123,0.913,0.9016,0.9294,0.9213,0.9312,0.94921,0.9443,0.9155,0.9463]
ALOI_data_LGCN_FF = [0.8572,0.8923, 0.9397,0.9103,0.9149,0.9557,0.9558,0.9587,0.9797,0.9512]
ALOI_data_JFGCN = [0.8603,0.9411,0.9468,0.9279,0.9499,0.9328,0.9478,0.9527,0.9588,0.9522]
ALOI_data_HGCN_MVSC= [0.8631,0.9311,0.9547,0.9615,0.9314,0.9627,0.9519,0.9554,0.9422,0.9432]
ALOI_data_Co_GCN =[0.8477,0.9312,0.9236,0.9278,0.9487,0.9567,0.9468,0.9567,0.9598,0.9612]
ALOI_data_DSIGCN = [0.8278,0.9412,0.9674,0.9517,0.9718,0.9536,0.9691,0.9718,0.9689,0.9792]
ALOI_data_GB = [0.9178,0.9662,0.9674,0.9517,0.9618,0.9736,0.9591,0.9818,0.9789,0.9892]


MNIST_data_GCN = [0.8027,0.8662,0.9043,0.9124,0.9237,0.9256,0.9278,0.9313,0.9366,0.9412]
MNIST_data_MLAN =[0.8023,0.8618,0.8816,0.8894,0.8913,0.9012,0.9121,0.9143,0.9156,0.923]
MNIST_data_DSRL =[0.8423,0.8728,0.8616,0.8694,0.9113,0.9112,0.9221,0.9243,0.9356,0.9323]
MNIST_data_LGCN_FF = [0.7772,0.8963, 0.9197,0.9203,0.9349,0.9357,0.9358,0.9367,0.9422,0.9498]
MNIST_data_JFGCN = [0.8203,0.9171,0.9168,0.9179,0.9289,0.9228,0.9378,0.9327,0.9421,0.9465]
MNIST_data_HGCN_MVSC= [0.8331,0.9082,0.9147,0.9215,0.9214,0.9327,0.9319,0.9454,0.9436,0.9596]
MNIST_data_Co_GCN =[0.8177,0.9022,0.9136,0.9178,0.9287,0.9267,0.9368,0.9367,0.9421,0.9399]
MNIST_data_DSIGCN = [0.8878,0.9372,0.9294,0.9356,0.9098,0.9221,0.9301,0.9433,0.9456,0.947]
MNIST_data_GB = [0.9018,0.9682,0.9694,0.9456,0.9598,0.9521,0.9601,0.9833,0.9856,0.977]

NUS_WIDE_data_GCN = [0.3127,0.3656,0.3743,0.3924,0.3937,0.4156,0.4278,0.4313,0.4412,0.4555]
NUS_WIDE_data_MLAN =[0.3023,0.3463,0.3416,0.3494,0.3513,0.3612,0.3621,0.3743,0.3799,0.3896]
NUS_WIDE_data_DSRL =[0.3023,0.4363,0.4456,0.4514,0.4932,0.5198,0.491,0.4863,0.4939,0.4926]
NUS_WIDE_data_LGCN_FF = [0.3672,0.4413, 0.4597,0.4603,0.4849,0.4857,0.4958,0.4987,0.5012,0.5521]
NUS_WIDE_data_JFGCN = [0.4203,0.4573,0.4678,0.4679,0.4789,0.4828,0.4878,0.4927,0.4921,0.5621]
NUS_WIDE_data_HGCN_MVSC= [0.4431,0.4926,0.4997,0.5115,0.5114,0.5527,0.5319,0.5454,0.5555,0.5632]
NUS_WIDE_data_Co_GCN =[0.2677,0.2752,0.2856,0.2978,0.3087,0.3167,0.3268,0.33670,0.3465,0.3566]
NUS_WIDE_data_DSIGCN = [0.4558,0.5422,0.5634,0.5197,0.5598,0.5886,0.5701,0.5918,0.593,0.5662]
NUS_WIDE_data_GB = [0.9258,0.97,0.9734,0.9797,0.9598,0.9786,0.9801,0.9718,0.983,0.9862]

BBCnews_data_GCN = [0.8027,0.8562,0.8643,0.8724,0.8737,0.8856,0.8978,0.9213,0.8923,0.9122]
BBCnews_data_MLAN =[0.6823,0.7412,0.7516,0.7894,0.7913,0.8012,0.8121,0.8143,0.8231,0.8621]
BBCnews_data_DSRL =[0.8223,0.9062,0.9116,0.9294,0.9313,0.9212,0.9321,0.9343,0.9131,0.9321]
BBCnews_data_LGCN_FF = [0.8372,0.9073, 0.9197,0.9203,0.9349,0.9357,0.9358,0.9387,0.9421,0.9489]
BBCnews_data_JFGCN = [0.8603,0.9323,0.9368,0.9479,0.9489,0.9528,0.9578,0.9627,0.9622,0.9669]
BBCnews_data_HGCN_MVSC= [0.9131,0.9636,0.9547,0.9615,0.9514,0.9627,0.9619,0.9654,0.9655,0.9621]
BBCnews_data_Co_GCN =[0.8077,0.9012,0.9236,0.9278,0.9187,0.9367,0.9468,0.9467,0.9521,0.9363]
BBCnews_data_DSIGCN = [0.8568,0.9122,0.9234,0.9217,0.9098,0.9546,0.8911,0.9368,0.9358,0.9687]
BBCnews_data_GB = [0.9568,0.9692,0.9634,0.9417,0.9598,0.9646,0.9711,0.9768,0.9758,0.9887]

BBCsports_data_GCN = [0.6027,0.6962,0.7043,0.7124,0.7237,0.7256,0.7578,0.7813,0.7932,0.8321]
BBCsports_data_MLAN =[0.5623,0.6268,0.6516,0.6894,0.6913,0.7012,0.7121,0.7143,0.7321,0.7555]
BBCsports_data_LGCN_FF = [0.9072,0.9713, 0.9797,0.9703,0.9749,0.9057,0.9758,0.9287,0.9621,0.9665]
BBCsports_data_JFGCN = [0.8803,0.9313,0.9568,0.9279,0.9389,0.9428,0.9478,0.9327,0.9698,0.9589]
BBCsports_data_HGCN_MVSC= [0.8631,0.9656,0.9547,0.9615,0.9614,0.9227,0.9319,0.9554,0.9663,0.9496]
BBCsports_data_Co_GCN =[0.9077,0.9432,0.9536,0.9578,0.9587,0.9367,0.9468,0.9567,0.9621,0.9498]
BBCsports_data_DSIGCN = [0.8758,0.9552,0.9314,0.9687,0.9418,0.9276,0.9471,0.9518,0.9499,0.9321]
BBCsports_data_GB = [0.9058,0.9672,0.9514,0.9687,0.9818,0.9476,0.9871,0.9618,0.9899,0.9721]

ORL_data_GCN = [0.3227,0.4421,0.4643,0.4724,0.5037,0.5156,0.5478,0.5813,0.5932,0.6221]
ORL_data_MLAN =[0.4623,0.4936,0.5116,0.5594,0.5313,0.5612,0.5821,0.6043,0.6321,0.6555]
ORL_data_DSRL =[0.4523,0.5026,0.4916,0.5094,0.5213,0.5412,0.5821,0.5643,0.5521,0.6255]
ORL_data_LGCN_FF = [0.4972,0.5586, 0.5997,0.6303,0.6649,0.6257,0.6758,0.7287,0.7821,0.7765]
ORL_data_JFGCN = [0.5503,0.6182,0.6368,0.6579,0.6689,0.6628,0.6778,0.6927,0.7098,0.7089]
ORL_data_HGCN_MVSC= [0.4631,0.5912,0.6547,0.7015,0.7214,0.7527,0.7719,0.7854,0.8163,0.8096]
ORL_data_Co_GCN =[0.4577,0.4812,0.5236,0.5578,0.5587,0.5367,0.5968,0.6267,0.6321,0.6698]
ORL_data_DSIGCN = [0.7158,0.7386,0.7514,0.7687,0.8118,0.8376,0.8371,0.8618,0.8899,0.8921]
ORL_data_GB = [0.7158,0.7386,0.7514,0.7687,0.8118,0.8376,0.8371,0.8618,0.8899,0.8921]

OutScene_data_GCN = [0.6027,0.6662,0.7043,0.7124,0.7237,0.7256,0.7378,0.7513,0.7621,0.7725]
OutScene_data_MLAN =[0.5223,0.5638,0.5516,0.5794,0.5913,0.6012,0.6121,0.6443,0.6821,0.6933]
OutScene_data_DSRL =[0.7221,0.75298,0.7616,0.7758,0.7996,0.8022,0.8269,0.8443,0.821,0.86966]
OutScene_data_LGCN_FF = [0.7772,0.7983, 0.8197,0.8203,0.8349,0.8357,0.8358,0.8487,0.8563,0.8635]
OutScene_data_JFGCN = [0.7203,0.7823,0.8068,0.8379,0.8289,0.8428,0.8478,0.85270,0.8621,0.8736]
OutScene_data_HGCN_MVSC= [0.7331,0.7956,0.8547,0.8615,0.8814,0.8927,0.9019,0.9054,0.9025,0.9155]
OutScene_data_Co_GCN =[0.7577,0.7612,0.7836,0.7978,0.8187,0.8367,0.8468,0.8567,0.8566,0.8645]
OutScene_data_DSIGCN = [0.6438,0.7622,0.7834,0.8607,0.8598,0.8286,0.85701,0.8719,0.8562,0.8599]
OutScene_data_GB = [0.8438,0.9622,0.9634,0.9507,0.9798,0.9686,0.9701,0.9519,0.9762,0.9799]


# for x, y in zip(x_axis_data, y_axis_data):
#     plt.text(x, y + 0.3, '%.00f' % y, ha='center', va='bottom', fontsize=7.5)  # y_axis_data1加标签数据

name = "NUS_WIDE"
plt.plot(x_axis_data, NUS_WIDE_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
plt.plot(x_axis_data, NUS_WIDE_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
plt.plot(x_axis_data, NUS_WIDE_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
plt.plot(x_axis_data, NUS_WIDE_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
plt.plot(x_axis_data, NUS_WIDE_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
plt.plot(x_axis_data, NUS_WIDE_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
plt.plot(x_axis_data, NUS_WIDE_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
plt.plot(x_axis_data, NUS_WIDE_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=2, label='MvRL-DP')
plt.plot(x_axis_data, NUS_WIDE_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

# name = "OutScene"
# plt.plot(x_axis_data, OutScene_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, OutScene_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, OutScene_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, OutScene_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, OutScene_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, OutScene_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, OutScene_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, OutScene_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=2, label='MvRL-DP')
# plt.plot(x_axis_data, OutScene_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

# name = "bbcnews"
# plt.plot(x_axis_data, BBCnews_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, BBCnews_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, BBCnews_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, BBCnews_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, BBCnews_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, BBCnews_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, BBCnews_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, BBCnews_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=1, label='MvRL-DP')
# plt.plot(x_axis_data, BBCnews_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

# name = "ORL"
# plt.plot(x_axis_data, ORL_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, ORL_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, ORL_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, ORL_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, ORL_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, ORL_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, ORL_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, ORL_data_DSIGCN, 'rx-', alpha=0.5, color='yellow',linewidth=2, label='MvRL-DP')
# plt.plot(x_axis_data, ORL_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

# plt.plot(x_axis_data, BBCsports_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, BBCsports_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, BBCsports_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, BBCsports_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, BBCsports_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, BBCsports_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, BBCsports_data_DSIGCN, 'rx-', alpha=0.8, color='yellow' ,linewidth=1, label='MvRL-DP')
# plt.plot(x_axis_data, BBCsports_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')
#
# name = "MNIST"
# plt.plot(x_axis_data, MNIST_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, MNIST_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, MNIST_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, MNIST_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, MNIST_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, MNIST_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, MNIST_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, MNIST_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=1, label='MvRL-DP')
# plt.plot(x_axis_data, MNIST_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')


# name = "ALOI"
#
# plt.plot(x_axis_data, ALOI_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, ALOI_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, ALOI_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, ALOI_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, ALOI_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, ALOI_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, ALOI_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, ALOI_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=1, label='MvRL-DP')
# plt.plot(x_axis_data, ALOI_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')
# plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

# ax = plt.subplot(111)
# #获取当前坐标轴的位置
# box = ax.get_position()
# #将坐标轴的位置上移10%
# ax.set_position([box.x0, box.y0 + box.height * 0.2,
#                  box.width, box.height * 0.9])
sns.set(style="darkgrid")  # 例如：seaborn-darkgrid, fivethirtyeight, ggplot 等

# 你的绘图代码
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")  # 显示上面的label
plt.xlabel('Label ratios', fontsize = '14')  # x_label
plt.yticks(np.arange(0, 1.05, 0.1))
plt.xticks(np.arange(0.05, 0.55, 0.05))
plt.legend(loc='lower right', markerscale=1.5)
plt.ylabel('ACC', fontsize = '14')  # y_label
plt.grid(True, linestyle="--", alpha=0.5)
y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# plt.yticks(y)
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig('./result/acc/'+name+'.png', dpi=1000)
plt.show()