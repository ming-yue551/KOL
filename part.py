# #分文件
# import pandas as pd
#
# # 配置参数
# input_file = "web-redditEmbeddings-users.csv"  # 替换为你的CSV文件名
# chunk_size = 24000
#
# # 分块读取并保存
# for idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size), 1):
#     output_file = f"web-redditEmbeddings-users_part_{idx}.csv"
#     chunk.to_csv(output_file, index=False, encoding="utf-8-sig")
#     print(f"已生成: {output_file}")

#------------------------------------------------------------------------------------------------------------
# #Reddit 数据特征对齐与矩阵构建
# import pandas as pd
# import numpy as np
#
# # --- 1. 加载 Reddit 投稿数据 (处理解析错误) ---
# try:
#     submissions = pd.read_csv('submissions.csv', on_bad_lines='skip', low_memory=False)
# except TypeError:
#     submissions = pd.read_csv('submissions.csv', error_bad_lines=False, warn_bad_lines=True)
#
# # 提取用户维度的统计特征
# user_stats = submissions.groupby('username').agg({
#     'score': 'mean',
#     'number_of_comments': 'sum',
#     'title': 'count'
# }).rename(columns={'title': 'post_count'}).reset_index()
#
# # --- 2. 加载 Embedding 数据 (处理无表头问题) ---
# # 注意：header=None 表示第一行不是表头，第一列是用户名
# embeddings = pd.read_csv('web-redditEmbeddings-users.csv', header=None)
#
# # 给第一列命名为 'username'，方便后续合并
# embeddings = embeddings.rename(columns={0: 'username'})
#
# # --- 3. 特征对齐合并 ---
# reddit_node_features = pd.merge(user_stats, embeddings, on='username', how='inner')
#
# print(f"成功对齐 Reddit 节点: {len(reddit_node_features)} 个")
# # 将结果保存，方便后续 GNN 使用
# reddit_node_features.to_csv('processed_reddit_nodes.csv', index=False)

# --------------------------------------------------------------------------------------------------------
# # 构建互动拓扑结构（生成“边”文件）
# import pandas as pd
# def build_bilibili_graph(file_path, output_name):
#     df = pd.read_csv(file_path)
#
#     # 建立 rpid 到 username 的映射表，方便通过 parent_rpid 找到被回复的人
#     rpid_to_user = dict(zip(df['rpid'], df['username']))
#
#     edges = []
#     for _, row in df.iterrows():
#         # 如果 parent_rpid 不为 0，说明这是一条回复
#         if row['parent_rpid'] != 0:
#             replier = row['username']
#             target = rpid_to_user.get(row['parent_rpid'])
#
#             # 确保目标存在且不是自言自语
#             if target and replier != target:
#                 edges.append({'source': replier, 'target': target, 'weight': 1})
#
#     edge_df = pd.DataFrame(edges)
#     # 按互动频率聚合边（如果 A 回复 B 多次，增加权重）
#     edge_df = edge_df.groupby(['source', 'target']).sum().reset_index()
#
#     edge_df.to_csv(f'{output_name}_edges.csv', index=False)
#     print(f"✅ {output_name} 的边文件已生成，共 {len(edge_df)} 条关系。")
#
#
# # 处理你的两个视频数据
# build_bilibili_graph('comments_BV1PyQzB7ER5_dream.csv', 'bili_dream')
# build_bilibili_graph('comments_BV1qm4y1r7BB_hair.csv', 'bili_hair')
#
# def extract_top_users_for_llm(file_path, output_name):
#     df = pd.read_csv(file_path)
#
#     # 筛选主评论（parent_rpid=0）且点赞数高的前 20 位用户
#     # 这些通常是潜在的意见领袖（KOL）
#     top_users = df[df['parent_rpid'] == 0].sort_values(by='like_count', ascending=False).head(20)
#
#     # 仅保留核心信息
#     llm_input = top_users[['username', 'content', 'like_count']]
#     llm_input.to_csv(f'{output_name}_for_llm.csv', index=False, encoding='utf-8-sig')
#     print(f"📝 已提取 {output_name} 的高赞评论，请用 LLM 进行打分。")
#
#
# extract_top_users_for_llm('comments_BV1PyQzB7ER5_dream.csv', 'dream')
# extract_top_users_for_llm('comments_BV1qm4y1r7BB_hair.csv', 'hair')

# ------------------------------------------------------------------------------------------------------------
# 将打分合并回去
import pandas as pd
import numpy as np

def process_and_align_data(video_name, llm_data, edges_file, reddit_file):
    print(f"--- 正在处理数据集: {video_name} ---")

    # 1. 加载数据
    # LLM 评分 (B站节点特征)
    df_llm = pd.DataFrame(llm_data, columns=['username', 'expertise', 'influence', 'sentiment'])

    # 社交边
    df_edges = pd.read_csv(edges_file)

    # Reddit 节点 (作为对比/预训练特征)
    df_reddit = pd.read_csv(reddit_file)

    # 2. 构建 B 站全局节点表
    # 从边文件中提取所有出现的用户名，确保没有遗漏
    all_bili_users = pd.unique(df_edges[['source', 'target']].values.ravel())
    nodes_bili = pd.DataFrame({'username': all_bili_users})

    # 合并 LLM 评分
    nodes_bili = pd.merge(nodes_bili, df_llm, on='username', how='left').fillna(0)

    # 3. 特征对齐 (核心步骤)
    # 因为 Reddit 有 300 维，B 站只有 3 维，我们需要统一维度才能输入同一个模型
    # 这里我们采用“对齐填充”策略：将 B 站特征扩展到与 Reddit 相同的列数
    reddit_feat_cols = [c for c in df_reddit.columns if
                        c not in ['username', 'score', 'number_of_comments', 'post_count']]
    target_dim = len(reddit_feat_cols)

    print(f"目标对齐维度: {target_dim}")

    # 为 B 站用户创建全零矩阵，并填入前三列特征
    bili_features = np.zeros((len(nodes_bili), target_dim))
    bili_features[:, 0] = nodes_bili['expertise']
    bili_features[:, 1] = nodes_bili['influence']
    bili_features[:, 2] = nodes_bili['sentiment']

    df_bili_final = pd.concat([nodes_bili['username'], pd.DataFrame(bili_features, columns=reddit_feat_cols)], axis=1)
    df_bili_final['platform'] = 'bilibili'

    # 提取 Reddit 核心特征并标记平台
    df_reddit_final = df_reddit[['username'] + reddit_feat_cols].copy()
    df_reddit_final['platform'] = 'reddit'

    # 4. 保存结果
    combined_nodes = pd.concat([df_bili_final, df_reddit_final], ignore_index=True)
    combined_nodes.to_csv(f'final_{video_name}_nodes.csv', index=False)
    df_edges.to_csv(f'final_{video_name}_edges.csv', index=False)

    print(f"✅ 完成！生成文件: final_{video_name}_nodes.csv (节点) 和 final_{video_name}_edges.csv (边)")
    print(f"总节点数: {len(combined_nodes)}, B站节点: {len(df_bili_final)}\n")

# --- 数据输入区 (根据 Gemini 的返回) ---
dream_llm = [
    ["崔听筠", 0.4, 0.9, 0.7], ["普瑞-赛斯_", 0.5, 0.8, 0.5], ["嘉泽胤阳", 0.2, 0.6, 0.3],
    ["薛定谔的狸花咕咕咕", 0.3, 0.5, 0.4], ["降萝伏莉仙帝", 0.6, 0.7, 0.6], ["M-V-T", 0.2, 0.4, 0.3],
    ["Rainman_E", 0.1, 0.5, 0.6], ["折纸无忧", 0.1, 0.4, 0.4], ["糸凛凛糸凛", 0.1, 0.3, 0.5], ["大乔617", 0.1, 0.3, 0.4]
]

hair_llm = [
    ["不知名花童", 0.9, 1.0, 0.8], ["䒴榆子", 0.7, 0.8, 0.6], ["乱码泛滥", 0.6, 0.9, 0.9],
    ["菇子怪", 0.3, 0.7, 0.8], ["木木牙叨叨", 0.4, 0.7, 0.9], ["不止不休", 0.5, 0.9, 1.0],
    ["接住我的小Yu滴", 0.4, 0.8, 0.8], ["长岛滚水", 0.6, 0.9, 0.9], ["bili_48364288435", 0.5, 0.8, 0.6]
]
# --- 执行处理 ---
process_and_align_data('dream', dream_llm, 'bili_dream_edges.csv', 'processed_reddit_nodes.csv')
process_and_align_data('hair', hair_llm, 'bili_hair_edges.csv', 'processed_reddit_nodes.csv')