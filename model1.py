import pandas as pd
import networkx as nx
import os


def generate_kol_report(nodes_path, edges_path, video_label):
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"跳过 {video_label}: 找不到文件")
        return

    # 1. 加载数据
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # 2. 构建有向图 (Directed Graph)
    # 在社交网络中，回复关系是有方向的：A -> B 表示 A 回复了 B，B 获得了影响力
    G = nx.DiGraph()

    # 添加所有节点
    for _, row in nodes_df.iterrows():
        G.add_node(row['username'])

    # 添加边并设置权重
    for _, row in edges_df.iterrows():
        # source 回复 target，则影响力由 source 流向 target
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    # 3. 算法计算
    # 计算 PageRank 值（代表节点在网络中的结构权重）
    pr_scores = nx.pagerank(G, weight='weight')

    # 计算入度（有多少人回复了他）
    in_degree = dict(G.in_degree())

    # 4. 融合 LLM 特征与结构权重
    report_data = []
    for user in G.nodes():
        user_info = nodes_df[nodes_df['username'] == user]

        if not user_info.empty:
            # 提取之前对齐好的前三列特征 (专业度, 影响力潜质, 情绪强度)
            # 注意：在 final_nodes 中，特征是从第 2 列到第 4 列
            expertise = user_info.iloc[0, 1]
            charisma = user_info.iloc[0, 2]
            sentiment = user_info.iloc[0, 3]
            platform = user_info.iloc[0, -1]  # 最后一列是平台标记

            # 获取结构评分
            struct_score = pr_scores.get(user, 0)
            engagement = in_degree.get(user, 0)

            # --- 核心计算公式 ---
            # 影响力 = (结构权重 * 权重系数) + (专业度 * 0.3) + (感召力 * 0.2)
            # 这里的 100 是为了将 PageRank 的小数值放大，方便观察
            influence_index = (struct_score * 100) * 0.5 + expertise * 0.3 + charisma * 0.2

            report_data.append({
                '用户名': user,
                '平台': platform,
                '综合影响力指数': round(influence_index, 4),
                '网络中心度(PageRank)': round(struct_score, 6),
                '被回复数(入度)': engagement,
                'LLM专业度': expertise,
                'LLM感召力': charisma
            })

    # 5. 排序并输出
    final_report = pd.DataFrame(report_data).sort_values(by='综合影响力指数', ascending=False)

    output_file = f'KOL_Rank_{video_label}.csv'
    final_report.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n--- {video_label} 意见领袖分析完成 ---")
    print(final_report[['用户名', '综合影响力指数', '网络中心度(PageRank)', 'LLM专业度']].head(10))
    print(f"完整报告已保存至: {output_file}")


# --- 执行分析 ---
generate_kol_report('final_dream_nodes.csv', 'final_dream_edges.csv', 'Dream_Video')
generate_kol_report('final_hair_nodes.csv', 'final_hair_edges.csv', 'PinkHair_Video')