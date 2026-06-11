import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def get_username_column(df):
    """自动检测用户名列名，支持 'username' 或 '用户名'"""
    if 'username' in df.columns:
        return 'username'
    elif '用户名' in df.columns:
        return '用户名'
    else:
        raise KeyError(f"未找到用户名列，实际列名: {df.columns.tolist()}")

def calculate_ndcg(pred_users, gt_dict, k):
    dcg = 0.0
    for i, user in enumerate(pred_users[:k]):
        rel = gt_dict.get(user, 0)
        dcg += (2**rel - 1) / np.log2(i + 2)
    ideal_rels = sorted(gt_dict.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / np.log2(i + 2)
    return dcg / (idcg + 1e-8)

def calculate_mrr(pred_users, gt_set):
    for i, user in enumerate(pred_users[:10]):
        if user in gt_set:
            return 1.0 / (i + 1)
    return 0.0

def calculate_kendall_tau(pred_users, gt_ordered, k):
    pred_top = pred_users[:k]
    gt_top = gt_ordered[:k]
    common = set(pred_top) & set(gt_top)
    if len(common) < 2:
        return 0.0
    pred_ranks = [pred_top.index(u) for u in common]
    gt_ranks = [gt_top.index(u) for u in common]
    tau, _ = kendalltau(pred_ranks, gt_ranks)
    return tau

def evaluate_variant(out_csv, truth_csv, topic_name):
    # 读取模型输出
    df_out = pd.read_csv(out_csv)
    user_col_out = get_username_column(df_out)
    df_out['username'] = df_out[user_col_out].astype(str).str.strip()
    pred_users = df_out.sort_values('GNN_Influence_Score', ascending=False)['username'].tolist()

    # 读取真值
    df_truth = pd.read_csv(truth_csv)
    user_col_truth = get_username_column(df_truth)
    df_truth['username'] = df_truth[user_col_truth].astype(str).str.strip()
    # 确定真值分数列（优先使用综合影响力指数，其次 PageRank，最后入度）
    if '综合影响力指数' in df_truth.columns:
        score_col = '综合影响力指数'
    elif '网络中心度(PageRank)' in df_truth.columns:
        score_col = '网络中心度(PageRank)'
    elif '被回复数(入度)' in df_truth.columns:
        score_col = '被回复数(入度)'
    else:
        raise KeyError(f"未找到分数列，实际列名: {df_truth.columns.tolist()}")
    df_truth['score'] = df_truth[score_col]
    # 构建真值字典
    gt_dict = dict(zip(df_truth['username'], df_truth['score']))
    min_s, max_s = df_truth['score'].min(), df_truth['score'].max()
    if max_s > min_s:
        gt_dict = {u: (v - min_s) / (max_s - min_s) * 10.0 for u, v in gt_dict.items()}
    else:
        gt_dict = {u: 0.0 for u in df_truth['username']}
    gt_sorted = df_truth.sort_values('score', ascending=False)['username'].tolist()
    gt_set = set(df_truth['username'])

    ndcg10 = calculate_ndcg(pred_users, gt_dict, 10)
    ndcg50 = calculate_ndcg(pred_users, gt_dict, 50)
    kendall50 = calculate_kendall_tau(pred_users, gt_sorted, 50)
    mrr10 = calculate_mrr(pred_users, gt_set)

    return {
        'Topic': topic_name,
        'NDCG@10': ndcg10,
        'NDCG@50': ndcg50,
        'KendallTau@50': kendall50,
        'MRR@10': mrr10
    }

if __name__ == '__main__':
    if not os.path.exists('ablation_file_list.csv'):
        print("未找到 ablation_file_list.csv，请先运行 run_ablation.py 生成各变体输出文件")
        exit(1)

    file_list = pd.read_csv('ablation_file_list.csv')
    all_results = []

    for _, row in file_list.iterrows():
        out_csv = row['out_csv']
        truth_csv = row['truth_csv']
        topic = row['topic']
        desc = row['desc']
        if not (os.path.exists(out_csv) and os.path.exists(truth_csv)):
            print(f"文件缺失: {out_csv} 或 {truth_csv}，跳过 {topic} - {desc}")
            continue
        print(f"正在评估 {topic} - {desc} ...")
        try:
            metrics = evaluate_variant(out_csv, truth_csv, topic)
            metrics['Variant'] = desc
            all_results.append(metrics)
        except Exception as e:
            print(f"评估 {topic} - {desc} 时出错: {e}")
            continue

    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n========== 细粒度消融实验结果 ==========")
        print(df_res[['Topic', 'Variant', 'NDCG@10', 'NDCG@50', 'KendallTau@50', 'MRR@10']].to_string(index=False))
        df_res.to_csv('ablation_fine_metrics.csv', index=False)
        print("\n结果已保存到 ablation_fine_metrics.csv")
    else:
        print("没有成功评估任何变体，请检查文件路径和列名。")