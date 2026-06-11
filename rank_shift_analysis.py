import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def get_username_column(df):
    """检测用户名列，支持 username 或 用户名"""
    if 'username' in df.columns:
        return 'username'
    elif '用户名' in df.columns:
        return '用户名'
    else:
        raise KeyError(f"未找到用户名列，实际列名: {df.columns.tolist()}")

def get_score_column(df):
    """检测分数列，优先综合影响力指数 -> 网络中心度 -> 被回复数"""
    if '综合影响力指数' in df.columns:
        return '综合影响力指数'
    elif '网络中心度(PageRank)' in df.columns:
        return '网络中心度(PageRank)'
    elif '被回复数(入度)' in df.columns:
        return '被回复数(入度)'
    else:
        raise KeyError(f"未找到分数列，实际列名: {df.columns.tolist()}")

def compute_rank_shift(pred_csv, truth_csv, k=50):
    """
    计算真值 Top-k 用户的排名位移
    返回: 平均绝对位移, 中位绝对位移, 以及各用户的位移列表
    """
    # 读取模型输出
    df_pred = pd.read_csv(pred_csv)
    user_col_pred = get_username_column(df_pred)
    df_pred['username'] = df_pred[user_col_pred].astype(str).str.strip()
    # 按模型分数降序得到排名列表
    df_pred = df_pred.sort_values('GNN_Influence_Score', ascending=False)
    pred_users = df_pred['username'].tolist()
    # 构建模型排名映射 {user: rank (1-indexed)}
    pred_rank = {user: i+1 for i, user in enumerate(pred_users)}

    # 读取真值
    df_truth = pd.read_csv(truth_csv)
    user_col_truth = get_username_column(df_truth)
    df_truth['username'] = df_truth[user_col_truth].astype(str).str.strip()
    score_col = get_score_column(df_truth)
    df_truth = df_truth.sort_values(score_col, ascending=False)
    truth_users = df_truth['username'].tolist()
    truth_rank = {user: i+1 for i, user in enumerate(truth_users)}

    # 取真值 Top-k
    topk_users = truth_users[:k]
    shifts = []
    for user in topk_users:
        if user in pred_rank:
            shift = abs(pred_rank[user] - truth_rank[user])
            shifts.append(shift)
        else:
            # 如果模型输出中找不到该用户（极少数情况），位移设为 k+1 作为惩罚
            shifts.append(k+1)
    return {
        'mean_shift': np.mean(shifts),
        'median_shift': np.median(shifts),
        'std_shift': np.std(shifts),
        'max_shift': max(shifts),
        'shifts': shifts
    }

def main():
    # 读取消融实验文件列表（由 run_ablation.py 生成）
    if not os.path.exists('ablation_file_list.csv'):
        print("未找到 ablation_file_list.csv，请先运行 run_ablation.py")
        return

    file_list = pd.read_csv('ablation_file_list.csv')
    results = []

    for _, row in file_list.iterrows():
        out_csv = row['out_csv']
        truth_csv = row['truth_csv']
        topic = row['topic']
        desc = row['desc']
        if not (os.path.exists(out_csv) and os.path.exists(truth_csv)):
            print(f"文件缺失: {out_csv} 或 {truth_csv}，跳过 {topic} - {desc}")
            continue
        print(f"正在计算 {topic} - {desc} 的排名位移...")
        metrics = compute_rank_shift(out_csv, truth_csv, k=50)
        results.append({
            'Topic': topic,
            'Variant': desc,
            'Mean Abs Shift (k=50)': metrics['mean_shift'],
            'Median Abs Shift': metrics['median_shift'],
            'Std Shift': metrics['std_shift'],
            'Max Shift': metrics['max_shift']
        })

    df_res = pd.DataFrame(results)
    print("\n========== 排名位移分析 (Top‑50 真值用户) ==========")
    print(df_res[['Topic', 'Variant', 'Mean Abs Shift (k=50)', 'Median Abs Shift', 'Std Shift', 'Max Shift']].to_string(index=False))
    df_res.to_csv('rank_shift_analysis.csv', index=False)
    print("\n结果已保存到 rank_shift_analysis.csv")

    # 额外输出一个简明的对比表格（只显示完整模型 vs 无对比学习）
    print("\n========== 重点对比：完整模型 vs 无对比学习 ==========")
    full = df_res[df_res['Variant'] == '完整模型'][['Topic', 'Mean Abs Shift (k=50)']].set_index('Topic')
    no_cl = df_res[df_res['Variant'] == '无对比学习'][['Topic', 'Mean Abs Shift (k=50)']].set_index('Topic')
    compare = full.join(no_cl, lsuffix='_full', rsuffix='_no_cl')
    print(compare)

if __name__ == '__main__':
    main()