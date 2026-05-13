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
        "GB-edge": [0.946, 0.935, 0.926, 0.921, 0.872],
        "KNN-edge": [0.939, 0.856, 0.775, 0.669, 0.624],
        "GB-feature": [0.966, 0.945, 0.944, 0.912, 0.899],
        "KNN-feature": [0.939, 0.812, 0.721, 0.665, 0.561],    # 新增线 1

    },
    "MNIST": {
        "GB-edge": [0.965, 0.934, 0.925, 0.915, 0.875],
        "KNN-edge": [0.91, 0.858, 0.835, 0.765, 0.638],
        "GB-feature": [0.965, 0.944, 0.931, 0.915, 0.892],
        "KNN-feature": [0.91, 0.852, 0.782, 0.731, 0.612],  # 新增线 1

    },
"NUS-WIDE": {
         "GB-edge": [0.964, 0.944, 0.921, 0.892, 0.884],
        "KNN-edge": [0.524, 0.468, 0.42, 0.415, 0.238],
        "GB-feature": [0.964, 0.9555, 0.928, 0.907, 0.862],
        "KNN-feature": [0.524, 0.4290, 0.385, 0.329, 0.272],    # 新增线 1

    },
"BBCnews": {
    "GB-edge": [0.966, 0.94, 0.921, 0.879, 0.865],
    "KNN-edge": [0.922, 0.858, 0.752, 0.715, 0.542],
    "GB-feature": [0.966,  0.952, 0.925, 0.911, 0.892],
    "KNN-feature": [0.922, 0.885, 0.815, 0.645, 0.632],  # 新增线 1

    },
"OutScene": {
         "GB-edge": [0.961, 0.921, 0.892, 0.894, 0.885],
        "KNN-edge": [0.785, 0.68, 0.712, 0.625, 0.538],
        "GB-feature": [0.961, 0.925, 0.928, 0.910, 0.932],
        "KNN-feature": [0.785, 0.710, 0.625, 0.579, 0.482],    # 新增线 1

    },
"ORL": {
         "GB-edge": [0.723, 0.714, 0.692, 0.658, 0.675],
        "KNN-edge": [0.67, 0.628, 0.562, 0.465, 0.328],
        "GB-feature": [0.723, 0.735, 0.718, 0.691, 0.692],
        "KNN-feature": [0.67, 0.555, 0.565, 0.519, 0.362],    # 新增线 1
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
    ax.set_ylabel('F1-score', fontsize=12)

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