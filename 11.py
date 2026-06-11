import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 学术期刊级绘图配置（与多子图脚本一致）
# ==========================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 600,
    'savefig.dpi': 600,
})

# 配色：绿（Dream）和橙（Hair）
DREAM_COLOR = '#6B828E'  # 灰绿色
HAIR_COLOR = '#BC907F'  # 陶土橙色
DREAM_FACE = '#9BA88F'  # 浅绿半透明
HAIR_FACE = '#D1B89C'  # 浅橙半透明

FILE_NAMES = {
    'Dream': 'KOL_GNN_Rank_Dream_Final_Optimized.csv',
    'Hair': 'KOL_GNN_Rank_Hair_Final_Optimized.csv'
}


def compute_stats(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    if mean > 0:
        sorted_scores = np.sort(scores)
        n = len(scores)
        gini = (np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_scores)) / (n * np.sum(sorted_scores) + 1e-8)
    else:
        gini = 0.0
    return mean, median, std, gini


def plot_single_raincloud(ax, data, label, color_main, color_face, jitter_width=0.12, point_size=3, alpha=0.4):
    """在给定的 axes 上绘制单个数据集的云雨图"""
    # 小提琴
    parts = ax.violinplot(data, positions=[1], showmeans=False, showmedians=False,
                          showextrema=False, widths=0.65)
    for pc in parts['bodies']:
        pc.set_facecolor(color_face)
        pc.set_edgecolor(color_main)
        pc.set_alpha(0.6)
        pc.set_linewidth(0.8)

    # 箱线图
    ax.boxplot(data, positions=[1], widths=0.2, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor='white', edgecolor=color_main, linewidth=0.8),
               medianprops=dict(color=color_main, linewidth=1.2),
               whiskerprops=dict(linewidth=0.6, color=color_main),
               capprops=dict(linewidth=0.6, color=color_main))

    # 抖动散点（zorder 较低，让标注浮在上面）
    values = np.array(data)
    x_vals = 1 + np.random.normal(0, jitter_width, size=len(values))
    ax.scatter(x_vals, values, s=point_size, alpha=alpha, color=color_main,
               edgecolors='none', rasterized=True, zorder=1)

    # 标注中位数（zorder 设为最高，置于顶层）
    median_val = np.median(data)
    ax.text(1, median_val + 0.02, f'median={median_val:.3f}', ha='center', va='bottom',
            fontsize=7, color=color_main, weight='bold', zorder=10)

    ax.set_xticks([1])
    ax.set_xticklabels([label], fontsize=11)
    # 纵轴范围 0~1.05，使图形居中（上下留白均匀）
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    # 读取数据
    dream_df = pd.read_csv(FILE_NAMES['Dream'])
    hair_df = pd.read_csv(FILE_NAMES['Hair'])
    dream_scores = dream_df['GNN_Influence_Score'].dropna().values
    hair_scores = hair_df['GNN_Influence_Score'].dropna().values

    # 统计指标
    dream_mean, dream_med, dream_std, dream_gini = compute_stats(dream_scores)
    hair_mean, hair_med, hair_std, hair_gini = compute_stats(hair_scores)

    print("\n========== 统计指标（完整模型 MDCE-GAT） ==========")
    print(f"Dream  : μ={dream_mean:.4f}, median={dream_med:.4f}, σ={dream_std:.4f}, Gini={dream_gini:.4f}")
    print(f"Hair   : μ={hair_mean:.4f}, median={hair_med:.4f}, σ={hair_std:.4f}, Gini={hair_gini:.4f}")
    print("=================================================\n")

    # 创建左右并排子图，不共享 y 轴（因为纵轴范围已统一设为 0~1.05）
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 5.0), sharey=False)

    # 左：Dream
    plot_single_raincloud(axes[0], dream_scores, 'Dream', DREAM_COLOR, DREAM_FACE,
                          jitter_width=0.1, point_size=3, alpha=0.4)
    axes[0].set_ylabel('MDCE-GAT Influence Score', fontsize=10)  # 注意：普通连字符 -
    text_dream = f"μ={dream_mean:.3f}\nGini={dream_gini:.3f}"
    axes[0].text(0.05, 0.98, text_dream, transform=axes[0].transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    # 右：Hair
    plot_single_raincloud(axes[1], hair_scores, 'Hair', HAIR_COLOR, HAIR_FACE,
                          jitter_width=0.1, point_size=3, alpha=0.4)
    text_hair = f"μ={hair_mean:.3f}\nGini={hair_gini:.3f}"
    axes[1].text(0.05, 0.98, text_hair, transform=axes[1].transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout(w_pad=2.0)
    plt.savefig('Fig6_Raincloud_split.png', dpi=600)
    plt.savefig('Fig6_Raincloud_split.svg', dpi=600)
    plt.savefig('Fig6_Raincloud_split.pdf', dpi=600)
    plt.show()
    print("✅ 已保存：纵轴 0~1.05，中位数标注置于顶层，纵轴标签使用普通连字符")


if __name__ == "__main__":
    main()