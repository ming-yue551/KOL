import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from math import pi
from matplotlib.colors import LinearSegmentedColormap

# --- 1. 莫兰迪三原色与高级色带 ---
COLORS = ["#9BA88F", "#6B828E", "#BC907F"]  # 灰绿(Model1)、石青(ModelB)、陶土(对比色)
# 构建学术风渐变
MY_CMP = LinearSegmentedColormap.from_list("morandi_sci", ["#F8F9FA", "#9BA88F", "#6B828E", "#BC907F"])

# --- 2. 全局学术化配置 ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True  # 自动调整子图间距

FILE_CONFIG = {
    'Dream': {
        'opt': 'KOL_GNN_Rank_Dream_Advanced.csv',
        'src': 'KOL_Rank_Dream_Video.csv',
        'edge': 'final_dream_edges.csv'
    },
    'Hair': {
        'opt': 'KOL_GNN_Rank_Hair_Final_Optimized.csv',
        'src': 'KOL_Rank_PinkHair_Video.csv',
        'edge': 'final_hair_edges.csv'
    }
}

# 子图配置：编号+标题+绘图函数映射
SUBPLOT_CONFIG = [
    ('a', 'LLM感召力-专业度四象限', 'plot_quadrant_sub'),
    ('b', '综合影响力vsGNN得分', 'plot_correlation_sub'),
    ('c', 'PageRank中心度分布', 'plot_pagerank_dist_sub'),
    ('d', '被回复数Top10用户', 'plot_top10_indegree_sub'),
    ('e', '专业度vs被回复数', 'plot_prof_indegree_sub'),
    ('f', 'GNN影响力得分分布', 'plot_gnn_score_dist_sub')
]


def load_and_clean(label, cfg):
    """加载并清洗数据（兼容双模型字段）"""
    df_opt = pd.read_csv(cfg['opt']).rename(columns={'用户名': 'username', 'username': 'username'})
    df_src = pd.read_csv(cfg['src']).rename(columns={'用户名': 'username', 'username': 'username'})
    edges = pd.read_csv(cfg['edge'])

    # 统一清洗用户名
    df_opt['username'] = df_opt['username'].astype(str).str.strip()
    df_src['username'] = df_src['username'].astype(str).str.strip()

    df = pd.merge(df_opt, df_src, on='username', how='inner')

    # 影响力分数归一化 (防止负值引起警告)
    if 'GNN_Influence_Score' in df.columns:
        s = df['GNN_Influence_Score']
        df['GNN_Influence_Score'] = (s - s.min()) / (s.max() - s.min())
    
    # Model1字段归一化（方便对比）
    if '综合影响力指数' in df.columns:
        s = df['综合影响力指数']
        df['综合影响力指数_归一化'] = (s - s.min()) / (s.max() - s.min())

    cols = ['GNN_Influence_Score', '网络中心度(PageRank)', '被回复数(入度)', 
            'LLM专业度', 'LLM感召力', '综合影响力指数', '综合影响力指数_归一化']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df, edges


# ------------------------------
# 6个子图的核心绘图函数（每个子图对应一个函数）
# ------------------------------
def plot_quadrant_sub(ax, df, label):
    """子图a：LLM感召力-专业度四象限（双模型气泡对比）"""
    m_x, m_y = df['LLM感召力'].mean(), df['LLM专业度'].mean()
    # 背景象限区分
    ax.axvspan(m_x, df['LLM感召力'].max(), m_y, df['LLM专业度'].max(), color=COLORS[0], alpha=0.05)
    ax.axhline(m_y, color='#999999', linestyle='--', lw=1)
    ax.axvline(m_x, color='#999999', linestyle='--', lw=1)
    
    # 气泡大小：ModelB(GNN)，颜色：Model1(综合影响力)
    bubble_size = df['GNN_Influence_Score'] * 800 + 50
    scatter = ax.scatter(df['LLM感召力'], df['LLM专业度'],
                         s=bubble_size, c=df['综合影响力指数_归一化'],
                         cmap=MY_CMP, alpha=0.75, edgecolors='white', linewidth=0.8)
    
    # Top3标注
    top_3 = df.nlargest(3, 'GNN_Influence_Score')
    for _, row in top_3.iterrows():
        ax.text(row['LLM感召力'], row['LLM专业度'] + 0.01, row['username'][:6], 
                fontsize=8, fontweight='bold')
    
    ax.set_xlabel('感召力', fontsize=10)
    ax.set_ylabel('专业度', fontsize=10)
    ax.set_title(f'{label} - 四象限定位', fontsize=11, pad=5)
    return scatter


def plot_correlation_sub(ax, df, label):
    """子图b：综合影响力(Model1) vs GNN得分(ModelB) 相关性"""
    # 散点+回归线（双模型对比核心）
    sns.regplot(data=df, x='综合影响力指数_归一化', y='GNN_Influence_Score',
                ax=ax, color=COLORS[1], scatter_kws={'alpha':0.6, 's':20, 'edgecolor':'white'},
                line_kws={'color':COLORS[2], 'lw':2})
    
    # 计算相关系数
    corr = df[['综合影响力指数_归一化', 'GNN_Influence_Score']].corr().iloc[0,1]
    ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Model1 综合影响力（归一化）', fontsize=10)
    ax.set_ylabel('ModelB GNN影响力得分', fontsize=10)
    ax.set_title(f'{label} - 双模型相关性', fontsize=11, pad=5)


def plot_pagerank_dist_sub(ax, df, label):
    """子图c：PageRank网络中心度分布（Model1核心指标）"""
    sns.histplot(data=df, x='网络中心度(PageRank)', ax=ax, 
                 color=COLORS[0], alpha=0.7, bins=20, kde=True)
    ax.axvline(df['网络中心度(PageRank)'].mean(), color=COLORS[2], 
               linestyle='--', lw=2, label='均值')
    ax.legend(fontsize=8)
    ax.set_xlabel('PageRank中心度', fontsize=10)
    ax.set_ylabel('频次', fontsize=10)
    ax.set_title(f'{label} - 网络中心度分布', fontsize=11, pad=5)


def plot_top10_indegree_sub(ax, df, label):
    """子图d：被回复数Top10用户（Model1互动指标）"""
    top10 = df.nlargest(10, '被回复数(入度)')[['username', '被回复数(入度)']]
    sns.barplot(data=top10, y='username', x='被回复数(入度)', ax=ax, color=COLORS[1], alpha=0.8)
    ax.set_xlabel('被回复数（入度）', fontsize=10)
    ax.set_ylabel('用户名', fontsize=10)
    ax.set_title(f'{label} - 互动量Top10', fontsize=11, pad=5)
    ax.tick_params(axis='y', labelsize=8)  # 缩小用户名字体


def plot_prof_indegree_sub(ax, df, label):
    """子图e：LLM专业度 vs 被回复数（Model1特征关联）"""
    sns.scatterplot(data=df, x='LLM专业度', y='被回复数(入度)', ax=ax,
                    color=COLORS[2], alpha=0.6, s=30, edgecolor='white')
    # 对数变换y轴（解决长尾分布）
    ax.set_yscale('log')
    ax.set_xlabel('LLM专业度', fontsize=10)
    ax.set_ylabel('被回复数（对数）', fontsize=10)
    ax.set_title(f'{label} - 专业度-互动量关联', fontsize=11, pad=5)


def plot_gnn_score_dist_sub(ax, df, label):
    """子图f：GNN影响力得分分布（ModelB核心指标）"""
    # 核密度对比：ModelB(GNN) vs Model1(综合影响力)
    sns.kdeplot(data=df, x='GNN_Influence_Score', ax=ax, 
                color=COLORS[1], label='ModelB(GNN)', fill=True, alpha=0.3)
    sns.kdeplot(data=df, x='综合影响力指数_归一化', ax=ax, 
                color=COLORS[0], label='Model1(综合)', fill=True, alpha=0.3)
    
    ax.legend(fontsize=8)
    ax.set_xlabel('归一化影响力得分', fontsize=10)
    ax.set_ylabel('密度', fontsize=10)
    ax.set_title(f'{label} - 双模型得分分布', fontsize=11, pad=5)


# ------------------------------
# 主函数：生成3×2六张子图的整合图
# ------------------------------
def plot_six_subplots(df, label, path):
    """生成3×2六张子图（带a-f编号+双模型对比）"""
    # 创建3×2子图网格，设置整体尺寸
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # 展平为一维数组，方便遍历

    # 遍历子图配置，逐个绘制
    for idx, (sub_id, sub_title, func_name) in enumerate(SUBPLOT_CONFIG):
        ax = axes[idx]
        # 调用对应子图函数
        plot_func = globals()[func_name]
        plot_func(ax, df, label)
        # 添加子图编号（左上角）
        ax.text(0.02, 0.98, f'{sub_id}) {sub_title}', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 统一添加全局标题（移除pad参数，改用y参数调整位置）
    fig.suptitle(f'KOL影响力分析 - {label} 数据集（双模型对比）', fontsize=16, y=0.98)

    # 调整整体布局，给顶部标题留出空间
    fig.subplots_adjust(top=0.92)

    # 添加统一的颜色条（对应子图a的颜色映射）
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # 颜色条位置：右+中
    scatter = axes[0].collections[0]  # 获取子图a的scatter对象
    fig.colorbar(scatter, cax=cbar_ax, label='Model1 综合影响力（归一化）')

    # 保存图片
    plt.savefig(f"{path}/Six_Subplots_{label}.png", bbox_inches='tight', dpi=300)
    plt.close()


# ------------------------------
# 原有函数保留（优化适配）
# ------------------------------
def plot_raincloud(all_results, path):
    """顶刊级云雨图：跨数据集双模型对比"""
    combined = []
    for lbl, df in all_results.items():
        df['Event'] = lbl
        # 新增模型标记列，方便对比
        df['Model1_Score'] = df['综合影响力指数_归一化']
        df['ModelB_Score'] = df['GNN_Influence_Score']
        combined.append(df)
    df_all = pd.concat(combined)

    plt.figure(figsize=(14, 8))
    # 双模型对比：上下分栏violin
    ax = sns.violinplot(data=df_all.melt(id_vars=['Event'], value_vars=['Model1_Score', 'ModelB_Score']),
                        x='Event', y='value', hue='variable',
                        palette=[COLORS[0], COLORS[1]], split=False, inner=None, alpha=0.2)

    sns.stripplot(data=df_all.melt(id_vars=['Event'], value_vars=['Model1_Score', 'ModelB_Score']),
                  x='Event', y='value', hue='variable',
                  palette=[COLORS[0], COLORS[1]], size=4, alpha=0.4, jitter=0.2, dodge=True)

    plt.title('Dream vs Hair: 双模型影响力分布对比', fontsize=16, pad=20)
    plt.ylabel('归一化影响力得分', fontsize=12)
    plt.xlabel('数据集', fontsize=12)
    plt.legend(title='模型', labels=['Model1(综合)', 'ModelB(GNN)'], fontsize=10)
    sns.despine(trim=True)
    plt.savefig(f"{path}/Raincloud_DoubleModel.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    out_dir = 'research_plots'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = {}
    for label, cfg in FILE_CONFIG.items():
        print(f"✨ 正在绘制 {label} 数据集的六子图分析...")
        sub_path = f"{out_dir}/{label}"
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # 加载清洗数据
        df, edges = load_and_clean(label, cfg)
        results[label] = df

        # 核心：生成3×2六张子图（带a-f编号+双模型对比）
        plot_six_subplots(df, label, sub_path)

    # 跨数据集双模型对比云雨图
    plot_raincloud(results, out_dir)
    print(f"\n✅ 所有图表已生成在 {out_dir} 目录下：")
    print(f"   - 每个数据集子目录下：Six_Subplots_{{Dream/Hair}}.png（3×2六张子图）")
    print(f"   - 根目录下：Raincloud_DoubleModel.png（跨数据集双模型对比）")