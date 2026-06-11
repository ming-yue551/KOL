import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def get_username_column(df):
    """检测用户名列名，支持 'username' 或 '用户名'"""
    if 'username' in df.columns:
        return 'username'
    elif '用户名' in df.columns:
        return '用户名'
    else:
        raise KeyError(f"未找到用户名列，实际列名: {df.columns.tolist()}")

def compute_degrees(edges_csv, nodes_csv):
    """从边文件中计算每个节点的总度数（入度+出度）"""
    edges = pd.read_csv(edges_csv)
    nodes = pd.read_csv(nodes_csv)
    # 统一用户名格式
    nodes['username'] = nodes['username'].astype(str).str.strip()
    # 统计每个用户的出现次数（作为source或target）
    all_users = pd.concat([edges['source'], edges['target']]).astype(str).str.strip()
    degree = all_users.value_counts().to_dict()
    # 给所有节点补0
    for user in nodes['username']:
        if user not in degree:
            degree[user] = 0
    return degree

def main():
    # 配置路径（请根据实际情况修改）
    topics = ['Dream', 'Hair']
    truth_files = {
        'Dream': 'KOL_Rank_Dream_Video.csv',
        'Hair': 'KOL_Rank_PinkHair_Video.csv'
    }
    edges_files = {
        'Dream': 'final_dream_edges.csv',
        'Hair': 'final_hair_edges.csv'
    }
    nodes_files = {
        'Dream': 'final_dream_nodes.csv',
        'Hair': 'final_hair_nodes.csv'
    }

    results = []
    for topic in topics:
        print(f"\n===== {topic} =====")
        # 读取真值
        df_truth = pd.read_csv(truth_files[topic])
        # 获取用户名列
        user_col = get_username_column(df_truth)
        df_truth['username'] = df_truth[user_col].astype(str).str.strip()
        # 确定分数列名
        if '综合影响力指数' in df_truth.columns:
            score_col = '综合影响力指数'
        elif '网络中心度(PageRank)' in df_truth.columns:
            score_col = '网络中心度(PageRank)'
        else:
            score_col = '被回复数(入度)'
        df_truth['truth_score'] = df_truth[score_col]

        # 计算度数
        degree_dict = compute_degrees(edges_files[topic], nodes_files[topic])
        df_truth['degree'] = df_truth['username'].map(degree_dict).fillna(0)

        # 计算相关系数
        pearson_r, pearson_p = pearsonr(df_truth['truth_score'], df_truth['degree'])
        spearman_r, spearman_p = spearmanr(df_truth['truth_score'], df_truth['degree'])

        print(f"分数列: {score_col}")
        print(f"Pearson 相关系数: {pearson_r:.4f} (p={pearson_p:.4e})")
        print(f"Spearman 相关系数: {spearman_r:.4f} (p={spearman_p:.4e})")

        results.append({
            'Topic': topic,
            'Score column': score_col,
            'Pearson r': pearson_r,
            'Spearman ρ': spearman_r,
            'N_nodes': len(df_truth)
        })

    print("\n========== 汇总 ==========")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()