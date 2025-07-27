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


ALOI_data_GCN = [0.8327,0.8562,0.8543,0.8624,0.8637,0.8756,0.8878,0.8813,0.8851,0.9066]
ALOI_data_MLAN =[0.8523,0.9138,0.8616,0.8694,0.8713,0.8812,0.8921,0.8943,0.9155,0.9063]
ALOI_data_DSRL =[0.7523,0.8618,0.8626,0.8794,0.8513,0.8912,0.91921,0.9143,0.9355,0.9263]
ALOI_data_LGCN_FF = [0.9372,0.9643, 0.9697,0.9503,0.9649,0.9657,0.9658,0.9787,0.9727,0.9712]
ALOI_data_JFGCN = [0.8903,0.9311,0.9368,0.9479,0.9499,0.9528,0.9678,0.9627,0.9688,0.9722]
ALOI_data_HGCN_MVSC= [0.8931,0.9461,0.9497,0.9515,0.9584,0.9627,0.9689,0.9714,0.9722,0.9822]
ALOI_data_Co_GCN =[0.8777,0.9112,0.9236,0.9285,0.9145,0.9287,0.9388,0.9497,0.9528,0.9632]
ALOI_data_DSIGCN = [0.83278,0.9412,0.9474,0.9567,0.9718,0.9536,0.9261,0.9508,0.9569,0.9692]
ALOI_data_GB = [0.9278,0.9662,0.9674,0.9667,0.9718,0.9736,0.9761,0.9808,0.9769,0.9892]


MNIST_data_GCN = [0.7927,0.8462,0.8543,0.8624,0.8639,0.8756,0.8878,0.8813,0.8866,0.8912]
MNIST_data_MLAN =[0.8423,0.8598,0.8616,0.8694,0.8713,0.8812,0.8921,0.9043,0.9156,0.9223]
MNIST_data_DSRL =[0.7923,0.8698,0.8716,0.8774,0.8833,0.8922,0.921,0.9143,0.9366,0.9223]
MNIST_data_LGCN_FF = [0.8272,0.8553, 0.8697,0.8703,0.8849,0.8957,0.9158,0.9267,0.9322,0.9398]
MNIST_data_JFGCN = [0.8603,0.9021,0.9068,0.9179,0.9189,0.9228,0.9278,0.9227,0.9321,0.9365]
MNIST_data_HGCN_MVSC= [0.8631,0.9012,0.9147,0.9215,0.9314,0.9327,0.9389,0.9414,0.9536,0.9596]
MNIST_data_Co_GCN =[0.8777,0.9012,0.9136,0.9169,0.9187,0.9227,0.9318,0.9417,0.9421,0.9499]
MNIST_data_DSIGCN = [0.89458,0.91202,0.9234,0.9361,0.9098,0.9221,0.9431,0.94833,0.9226,0.931]
MNIST_data_GB = [0.9458,0.9602,0.9634,0.9761,0.9798,0.9821,0.9831,0.9833,0.9826,0.987]

NUS_WIDE_data_GCN = [0.3127,0.3556,0.3763,0.3912,0.4037,0.4056,0.4128,0.4193,0.4212,0.4235]
NUS_WIDE_data_MLAN =[0.3023,0.3323,0.3456,0.3514,0.3632,0.3698,0.3751,0.3863,0.3939,0.3926]
NUS_WIDE_data_DSRL =[0.4023,0.4323,0.4326,0.4514,0.4632,0.448,0.461,0.4863,0.4639,0.5326]
NUS_WIDE_data_LGCN_FF = [0.3672,0.4413, 0.4597,0.4658,0.4895,0.4836,0.5028,0.4987,0.5012,0.5121]
NUS_WIDE_data_JFGCN = [0.4203,0.4463,0.4678,0.4679,0.4789,0.4828,0.4878,0.4927,0.4921,0.5321]
NUS_WIDE_data_HGCN_MVSC= [0.4031,0.4636,0.4997,0.5115,0.5294,0.5527,0.5419,0.5454,0.5565,0.5732]
NUS_WIDE_data_Co_GCN =[0.2577,0.2812,0.2956,0.2978,0.3087,0.3155,0.3241,0.3362,0.3565,0.3667]
NUS_WIDE_data_DSIGCN = [0.41258,0.5633,0.5534,0.5697,0.6398,0.6286,0.6501,0.5918,0.6541,0.6652]
NUS_WIDE_data_GB = [0.9158,0.9552,0.9534,0.9697,0.9798,0.9886,0.9801,0.9918,0.9941,0.9952]

BBCnews_data_GCN = [0.7627,0.8465,0.8463,0.8624,0.8737,0.8856,0.8978,0.9213,0.8923,0.9122]
BBCnews_data_MLAN =[0.6823,0.7352,0.7516,0.7694,0.7813,0.7912,0.8021,0.8043,0.8131,0.8321]
BBCnews_data_DSRL =[0.7823,0.912,0.9216,0.9294,0.9313,0.9412,0.9421,0.9243,0.9331,0.9321]
BBCnews_data_LGCN_FF = [0.8282,0.9043, 0.9197,0.9212,0.9349,0.939,0.9358,0.9487,0.9521,0.9589]
BBCnews_data_JFGCN = [0.8503,0.9253,0.9368,0.9379,0.9489,0.9428,0.9578,0.9517,0.9492,0.9569]
BBCnews_data_HGCN_MVSC= [0.9231,0.9656,0.9647,0.9615,0.9714,0.9727,0.9619,0.9754,0.9755,0.9812]
BBCnews_data_Co_GCN =[0.8077,0.8932,0.9236,0.9278,0.9187,0.9367,0.9468,0.9467,0.9521,0.9363]
BBCnews_data_DSIGCN = [0.8428,0.9012,0.9134,0.9217,0.9298,0.9446,0.9411,0.9628,0.9358,0.9487]
BBCnews_data_GB = [0.9428,0.9662,0.9634,0.9617,0.9698,0.9746,0.9711,0.9728,0.9758,0.9887]

BBCsports_data_GCN = [0.5527,0.6812,0.6943,0.7024,0.7157,0.7236,0.7468,0.7593,0.7732,0.8121]
BBCsports_data_MLAN =[0.547,0.6268,0.6516,0.6754,0.6863,0.7102,0.7191,0.7143,0.7321,0.7565]
BBCsports_data_DSRL =[0.547,0.6268,0.6516,0.6754,0.6863,0.7102,0.7191,0.7143,0.7321,0.7565]
BBCsports_data_LGCN_FF = [0.9272,0.9703, 0.9797,0.9768,0.9749,0.9857,0.9758,0.9887,0.9921,0.9965]
BBCsports_data_JFGCN = [0.8503,0.9321,0.9468,0.9579,0.9689,0.9689,0.9678,0.9668,0.9718,0.9789]
BBCsports_data_HGCN_MVSC= [0.9031,0.9456,0.9558,0.9628,0.9668,0.9727,0.9769,0.9854,0.9833,0.9896]
BBCsports_data_Co_GCN =[0.8577,0.9342,0.926,0.9478,0.9487,0.9367,0.9468,0.9567,0.9621,0.9698]
BBCsports_data_GB = [0.9368,0.9542,0.9904,0.9877,0.9818,0.9746,0.9871,0.9838,0.9899,0.9851]

OutScene_data_GCN = [0.6028,0.6512,0.6843,0.7124,0.7217,0.7255,0.7378,0.7513,0.7681,0.7765]
OutScene_data_MLAN =[0.5223,0.5498,0.5616,0.5658,0.5996,0.6022,0.6969,0.6743,0.6821,0.6966]
OutScene_data_DSRL =[0.6523,0.7498,0.7616,0.7758,0.7996,0.8022,0.8269,0.8243,0.7821,0.8566]
OutScene_data_LGCN_FF = [0.7272,0.7953, 0.8197,0.8003,0.8349,0.8457,0.8458,0.8587,0.8563,0.8635]
OutScene_data_JFGCN = [0.6603,0.7763,0.8368,0.8679,0.8789,0.8728,0.8878,0.90270,0.9121,0.9116]
OutScene_data_HGCN_MVSC= [0.6331,0.7926,0.8247,0.8315,0.8514,0.8627,0.8719,0.8954,0.8925,0.9155]
OutScene_data_Co_GCN =[0.7277,0.7622,0.7736,0.7878,0.8087,0.8267,0.8325,0.8457,0.8566,0.8612]
OutScene_data_GB = [0.9118,0.9592,0.9634,0.9297,0.9298,0.9566,0.9701,0.9529,0.9562,0.9539]
OutScene_data_DSIGCN = [0.6418,0.7722,0.75634,0.7597,0.8198,0.7966,0.8701,0.8329,0.8562,0.8539]

ORL_data_GCN = [0.2527,0.3214,0.3543,0.3624,0.3837,0.4156,0.4478,0.4493,0.4132,0.4521]
ORL_data_MLAN =[0.3523,0.3796,0.3916,0.4094,0.4213,0.4412,0.4821,0.4643,0.4521,0.4555]
ORL_data_DSRL =[0.4523,0.5296,0.4916,0.5394,0.5613,0.5812,0.5921,0.5843,0.6121,0.6655]
ORL_data_LGCN_FF = [0.5172,0.5651, 0.5897,0.6503,0.6949,0.7057,0.6858,0.7187,0.7221,0.7065]
ORL_data_JFGCN = [0.4803,0.5862,0.6568,0.6679,0.6889,0.7128,0.7578,0.7427,0.7698,0.7589]
ORL_data_HGCN_MVSC= [0.5631,0.5892,0.6147,0.6315,0.6614,0.6727,0.7219,0.7554,0.7263,0.7396]
ORL_data_Co_GCN =[0.4077,0.4821,0.5136,0.5378,0.5287,0.5967,0.6568,0.6667,0.6621,0.6698]
ORL_data_GB = [0.7058,0.7236,0.7514,0.7687,0.7818,0.7876,0.8071,0.8218,0.8199,0.8121]
ORL_data_DSIGCN = [0.5558,0.6316,0.6514,0.6687,0.6818,0.6876,0.6571,0.7018,0.7599,0.7221]

# for x, y in zip(x_axis_data, y_axis_data):
#     plt.text(x, y + 0.3, '%.00f' % y, ha='center', va='bottom', fontsize=7.5)  # y_axis_data1加标签数据

# name = "NUS_WIDE"
# plt.plot(x_axis_data, NUS_WIDE_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
# plt.plot(x_axis_data, NUS_WIDE_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
# plt.plot(x_axis_data, NUS_WIDE_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
# plt.plot(x_axis_data, NUS_WIDE_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
# plt.plot(x_axis_data, NUS_WIDE_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
# plt.plot(x_axis_data, NUS_WIDE_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
# plt.plot(x_axis_data, NUS_WIDE_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
# plt.plot(x_axis_data, NUS_WIDE_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=2, label='MvRL-DP')
# plt.plot(x_axis_data, NUS_WIDE_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

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


name = "ALOI"

plt.plot(x_axis_data, ALOI_data_GCN, 'y*-', alpha=0.5, linewidth=2, label='GCN')
plt.plot(x_axis_data, ALOI_data_MLAN, 'g+-', alpha=0.5, linewidth=2, label='MLAN')
plt.plot(x_axis_data, ALOI_data_DSRL, 'b+-', alpha=0.5, linewidth=2, label='DSRL')
plt.plot(x_axis_data, ALOI_data_LGCN_FF, 'ms-', alpha=0.5, linewidth=2, label='LGCN_FF')
plt.plot(x_axis_data, ALOI_data_Co_GCN, 'd-',color='darkslategray', alpha=0.5, linewidth=2, label='Co-GCN')
plt.plot(x_axis_data, ALOI_data_JFGCN, 'cP-', color='orange',alpha=0.5, linewidth=2, label='JFGCN')
plt.plot(x_axis_data, ALOI_data_HGCN_MVSC, '<-', color='gold',alpha=0.5, linewidth=2, label='HGCN_MVSC')
plt.plot(x_axis_data, ALOI_data_DSIGCN, 'rx-', alpha=0.5, color='yellow' ,linewidth=1, label='MvRL-DP')
plt.plot(x_axis_data, ALOI_data_GB, 'rs-', alpha=0.5, linewidth=4, label='GBOC-GCN')

# ax = plt.subplot(111)
# #获取当前坐标轴的位置
# box = ax.get_position()
# #将坐标轴的位置上移10%
# ax.set_position([box.x0, box.y0 + box.height * 0.2,
#                  box.width, box.height * 0.9])
sns.set(style="darkgrid")
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")  # 显示上面的label
plt.xlabel('Label ratios',fontsize = '14')  # x_label
plt.yticks(np.arange(0.5, 1.05, 0.1))
plt.xticks(np.arange(0.05, 0.55, 0.05))
plt.legend(loc='lower right', markerscale=1.5)
plt.ylabel('F1-score',fontsize = '14')  # y_label
plt.grid(True, linestyle="--", alpha=0.5)
y = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# plt.yticks(y)
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig('./result/f1/'+name+'.png', dpi=1000)
plt.show()