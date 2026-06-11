import pandas as pd

# 读取数据（请修改为您的实际路径）
edges = pd.read_csv('final_hair_edges.csv')
nodes = pd.read_csv('final_hair_nodes.csv')

# 节点总数
num_nodes = len(nodes)

# 有向加权边数
num_edges = len(edges)

# 网络密度（有向图）：|E| / (|V|*(|V|-1))
density = num_edges / (num_nodes * (num_nodes - 1))

# 计算每个节点的总度数（入度+出度）
all_users = pd.concat([edges['source'], edges['target']]).astype(str).str.strip()
degree = all_users.value_counts()
# 零度节点：在节点文件中但度数为0的节点
zero_degree_nodes = num_nodes - len(degree)

# 最大边权（如果边文件有 weight 列）
if 'weight' in edges.columns:
    max_weight = edges['weight'].max()
    mean_weight = edges['weight'].mean()
else:
    # 如果没有 weight 列，则每条边计为1
    max_weight = 1
    mean_weight = 1

print(f"User node count: {num_nodes}")
print(f"Directed weighted edge count: {num_edges}")
print(f"Network density: {density:.2e}")
print(f"Average outdegree: {num_edges / num_nodes:.2f}")
print(f"Number of non-isolated nodes: {len(degree)}")
print(f"Number of zero-degree nodes: {zero_degree_nodes}")
print(f"Maximum edge weight: {max_weight}")
print(f"Mean edge weight: {mean_weight:.2f}")