import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# SCI 顶刊级学术标准绘图配置 (IEEE/ACM 双栏规范)
# ==========================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.titlesize': 9,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
})

COLORS = ["#9BA88F", "#6B828E", "#BC907F"]
MY_CMP = LinearSegmentedColormap.from_list("sci_cmap", ["#F8F9FA", "#9BA88F", "#6B828E", "#BC907F"])

COLUMN_MAPPER = {
    '用户名': 'username',
    '网络中心度(PageRank)': 'pagerank',
    '被回复数(入度)': 'indegree',
    'LLM专业度': 'professionalism',
    'LLM感召力': 'appeal',
    '综合影响力指数': 'comprehensive_norm'
}

FILE_CONFIG = {
    'Dream': {
        'opt': 'KOL_GNN_Rank_Dream_Final_Optimized.csv',
        'src': 'KOL_Rank_Dream_Video.csv'
    },
    'Hair': {
        'opt': 'KOL_GNN_Rank_Hair_Final_Optimized.csv',
        'src': 'KOL_Rank_PinkHair_Video.csv'
    }
}

SUBPLOT_CONFIG = [
    ('a', 'Content Quality Synergy Analysis', 'plot_quadrant_sub'),
    ('b', 'Dual-Model Decision Correlation', 'plot_correlation_sub'),
    ('c', 'Topology Centrality Density Distribution', 'plot_pagerank_dist_sub'),
    ('d', 'Structural Virality Elite Users', 'plot_top10_indegree_sub'),
    ('e', 'Expertise vs. Network Reception', 'plot_prof_indegree_sub'),
    ('f', 'Probability Distribution Shift Comparison', 'plot_gnn_score_dist_sub')
]


def load_and_clean(label, cfg):
    df_opt = pd.read_csv(cfg['opt']).rename(columns=COLUMN_MAPPER)
    df_src = pd.read_csv(cfg['src']).rename(columns=COLUMN_MAPPER)

    df_opt['username'] = df_opt['username'].astype(str).str.strip()
    df_src['username'] = df_src['username'].astype(str).str.strip()

    df = pd.merge(df_opt, df_src, on='username', how='inner')

    if 'comprehensive_norm' in df.columns:
        s = df['comprehensive_norm']
        df['comprehensive_norm'] = (s - s.min()) / (s.max() - s.min() + 1e-8)

    cols = ['GNN_Influence_Score', 'pagerank', 'indegree', 'professionalism', 'appeal', 'comprehensive_norm']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


# ==========================================
# 子图绘制引擎（包含隐私脱敏与格式修复）
# ==========================================
def plot_quadrant_sub(ax, df):
    x = df['appeal']
    y = df['professionalism']
    ax.axhline(y.mean(), color='#BDC3C7', linestyle='--', lw=0.5)
    ax.axvline(x.mean(), color='#BDC3C7', linestyle='--', lw=0.5)

    scatter = ax.scatter(x, y, c=df['comprehensive_norm'],
                         s=df['GNN_Influence_Score'] * 75 + 8,
                         cmap=MY_CMP, alpha=0.75, edgecolors='#7F8C8D', linewidth=0.3)
    ax.set_xlabel('LLM Measured Appeal')
    ax.set_ylabel('LLM Measured Professionalism')

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Model1 Composite Norm', fontsize=6)


def plot_correlation_sub(ax, df):
    sns.regplot(x='comprehensive_norm', y='GNN_Influence_Score', data=df,
                ax=ax, color=COLORS[1],
                scatter_kws={'s': 8, 'alpha': 0.5, 'edgecolors': 'none'},
                line_kws={'lw': 0.8, 'color': '#C0392B'})
    corr = df[['comprehensive_norm', 'GNN_Influence_Score']].corr().iloc[0, 1]
    ax.text(0.06, 0.90, f'Pearson $r$ = {corr:.3f}', transform=ax.transAxes,
            fontsize=7, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    ax.set_xlabel('Model1 Index (Normalized)')
    ax.set_ylabel('MDCE-GAT Influence Score')


def plot_pagerank_dist_sub(ax, df):
    sns.histplot(df['pagerank'], bins=18, color=COLORS[0], alpha=0.6, kde=True,
                 ax=ax, line_kws={'lw': 0.8}, edgecolor='white', linewidth=0.2)
    ax.axvline(df['pagerank'].mean(), color=COLORS[2], lw=0.8, linestyle='--')
    ax.set_xlabel('PageRank Centrality Value')
    ax.set_ylabel('Frequency Count')


def plot_top10_indegree_sub(ax, df):
    # 🌟 核心修复：截取前 8 位大 V，并按照入度从低到高排序以便在条形图中从上往下合理呈现
    top = df.nlargest(8, 'indegree').sort_values(by='indegree', ascending=True)

    # 🌟 隐私安全与显示修复：抛弃真实用户名，改用学术界通用的 'Leader 1' - 'Leader 8' 代号
    # 按照入度大小赋予代号（第一名是 Leader 1，条形图最长，排在最上方）
    rank_indices = range(len(top), 0, -1)
    top['user_label'] = [f'Leader {r}' for r in rank_indices]

    ax.barh(top['user_label'], top['indegree'], color=COLORS[1], height=0.6, edgecolor='none')
    ax.set_xlabel('In-Degree Count')
    ax.set_ylabel('Identified Key Users (Anonymized)')

    # 🌟 刻度参数微调：增加留白空间，确保 'Leader X' 的纯英文标签能 100% 完整显示
    ax.tick_params(axis='y', pad=6, labelsize=7.5)
    ax.grid(axis='x', linestyle=':', alpha=0.5, lw=0.4)


def plot_prof_indegree_sub(ax, df):
    ax.scatter(df['professionalism'], df['indegree'] + 1,
               color=COLORS[2], s=8, alpha=0.5, edgecolors='#7F8C8D', linewidth=0.2)
    ax.set_yscale('log')
    ax.set_xlabel('LLM Professionalism Dimension')
    ax.set_ylabel('In-Degree Level (Log Scale)')


def plot_gnn_score_dist_sub(ax, df):
    sns.kdeplot(df['GNN_Influence_Score'], color='#C0392B', fill=True, alpha=0.2, ax=ax, label='MDCE-GAT (Ours)',
                lw=0.8)
    sns.kdeplot(df['comprehensive_norm'], color=COLORS[0], fill=True, alpha=0.15, ax=ax, label='Baseline Model', lw=0.8)
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlabel('Latent Influence Score')
    ax.set_ylabel('Probability Density')


# ==========================================
# 主控制台
# ==========================================
def plot_sci_figure(df, label, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5))
    axes = axes.flatten()

    for i, (pid, title, func) in enumerate(SUBPLOT_CONFIG):
        ax = axes[i]
        globals()[func](ax, df)

        ax.set_title(title, pad=6, weight='bold', color='#2C3E50')
        ax.text(-0.15, 1.08, f'({pid})', transform=ax.transAxes, fontsize=9, va='top', weight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(width=0.5, length=3)

    plt.tight_layout(w_pad=1.5, h_pad=2.0)

    tiff_output = f'{save_path}/Figure_{label}_MDCE_GAT.tiff'
    pdf_output = f'{save_path}/Figure_{label}_MDCE_GAT.pdf'
    png_output = f'{save_path}/Figure_{label}_MDCE_GAT.png'

    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.savefig(tiff_output, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_output, bbox_inches='tight')
    plt.close()
    print(f"   >> [{label}] 体系子图安全重构成功 -> 已同步生成匿名脱敏版 PNG")


if __name__ == "__main__":
    out_dir = 'SCI_Figures'
    os.makedirs(out_dir, exist_ok=True)

    print("🚀 启动大创/SCI期刊标准高分辨【匿名隐私保护版】图表渲染流水线...")
    for name, config in FILE_CONFIG.items():
        if os.path.exists(config['opt']) and os.path.exists(config['src']):
            df = load_and_clean(name, config)
            os.makedirs(f'{out_dir}/{name}', exist_ok=True)
            plot_sci_figure(df, name, f'{out_dir}/{name}')
        else:
            print(f"⚠️ 无法读取 {name} 的核心 CSV，请确认文件名无误。")