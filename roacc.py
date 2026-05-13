# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 全局样式配置
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "black"})

# 横坐标与模拟数据 (请替换为你实际的实验数据)
x_axis_data = [0.0, 2.0, 4.0, 6.0, 8.0]
datasets_data = {
    "ALOI": {
        "GB-edge": [0.966, 0.95, 0.96, 0.921, 0.852],
        "KNN-edge": [0.945, 0.84, 0.75, 0.69, 0.64],
        "GB-feature": [0.966, 0.955, 0.944, 0.912, 0.899],
        "KNN-feature": [0.945, 0.82, 0.751, 0.665, 0.561],    # 新增线 1

    },
    "MNIST": {
        "GB-edge": [0.968, 0.94, 0.925, 0.895, 0.855],
        "KNN-edge": [0.903, 0.88, 0.855, 0.775, 0.628],
        "GB-feature": [0.968, 0.944, 0.931, 0.915, 0.892],
        "KNN-feature": [0.903, 0.850, 0.812, 0.711, 0.612],  # 新增线 1

    },
"NUS-WIDE": {
         "GB-edge": [0.97, 0.94, 0.891, 0.89, 0.84],
        "KNN-edge": [0.559, 0.48, 0.42, 0.415, 0.238],
        "GB-feature": [0.97, 0.9555, 0.878, 0.870, 0.862],
        "KNN-feature": [0.559, 0.490, 0.485, 0.379, 0.272],    # 新增线 1

    },
"BBCnews": {
    "GB-edge": [0.969, 0.94, 0.921, 0.879, 0.855],
    "KNN-edge": [0.941, 0.88, 0.812, 0.715, 0.622],
    "GB-feature": [0.969,  0.952, 0.925, 0.911, 0.892],
    "KNN-feature": [0.941, 0.85, 0.815, 0.675, 0.632],  # 新增线 1

    },
"OutScene": {
         "GB-edge": [0.962, 0.901, 0.882, 0.894, 0.855],
        "KNN-edge": [0.812, 0.78, 0.712, 0.675, 0.538],
        "GB-feature": [0.962, 0.925, 0.928, 0.910, 0.892],
        "KNN-feature": [0.812, 0.790, 0.685, 0.579, 0.512],    # 新增线 1

    },
"ORL": {
         "GB-edge": [0.738, 0.724, 0.692, 0.68, 0.675],
        "KNN-edge": [0.683, 0.688, 0.612, 0.515, 0.468],
        "GB-feature": [0.738, 0.735, 0.718, 0.691, 0.692],
        "KNN-feature": [0.683, 0.59, 0.585, 0.519, 0.472],    # 新增线 1
    },
    # ... 其他数据集 (NUS-WIDE, BBCnews, OutScene, ORL) 依此类推
}

# 2. 线条样式定义
styles = {
    # -edge 统一使用：红色 (#e66b6b)，方块标记 (s)
    'GB-edge':   {'fmt': 'rs-',  'color': '#e66b6b', 'lw': 3.0, 'ms': 6, 'label': 'GB-edge'},    # GB实线
    'KNN-edge':  {'fmt': 'rs--', 'color': '#e66b6b', 'lw': 2.0, 'ms': 6, 'label': 'KNN-edge'},   # KNN虚线

    # -feature 统一使用：蓝色 (#5d8ecf)，圆点标记 (o)
    'GB-feature':  {'fmt': 'bo-',  'color': '#5d8ecf', 'lw': 2.0, 'ms': 5, 'label': 'GB-feature'}, # GB实线
    'KNN-feature': {'fmt': 'bo--', 'color': '#5d8ecf', 'lw': 1.5, 'ms': 5, 'label': 'KNN-feature'} # KNN虚线
}
plot_configs = [
    {"name": "ALOI", "title": "(a) ALOI"},
    {"name": "MNIST", "title": "(b) MNIST"},
    {"name": "NUS-WIDE", "title": "(c) NUS-WIDE"},
    {"name": "BBCnews", "title": "(d) BBCnews"},
    {"name": "OutScene", "title": "(e) OutScene"},
    {"name": "ORL", "title": "(f) ORL"}
]

# 3. 开始绘图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

lines_for_legend = []
labels_for_legend = []

for i, config in enumerate(plot_configs):
    ax = axes[i]
    dataset = config["name"]
    curr_data = datasets_data[dataset]

    # 绘制折线
    for model in ['GB-edge', 'KNN-edge', 'GB-feature', 'KNN-feature']:
        s = styles[model]
        line, = ax.plot(x_axis_data, curr_data[model], s['fmt'],
                        color=s['color'], linewidth=s['lw'],
                        markersize=s['ms'], label=s['label'])
        if i == 0:
            lines_for_legend.append(line)
            labels_for_legend.append(s['label'])

    # 坐标轴基础设置
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_axis_data)
    ax.set_xlabel('Addition rate', fontsize=12)
    ax.set_ylabel('ACC', fontsize=12)

    # --- 关键修改：数据集名称放到子图下方 ---
    # 使用 ax.text 将标题定位在子图坐标系之外的下方
    ax.text(0.5, -0.28, config["title"], transform=ax.transAxes,
            fontsize=14,  ha='center', va='top')

# 4. 全局共享图例 (保持在顶部横向排列)
fig.legend(lines_for_legend, labels_for_legend,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.98),
           ncol=5,
           frameon=False,
           fontsize=13,
           columnspacing=2.0)

# 5. 调整整体布局
# 增加 hspace 以确保第一排的下方标题不会遮挡第二排子图
plt.subplots_adjust(wspace=0.25, hspace=0.5, top=0.9, bottom=0.15)
# 保存和显示
plt.savefig('./robustness_results_final.png', dpi=600, bbox_inches='tight')
plt.show()