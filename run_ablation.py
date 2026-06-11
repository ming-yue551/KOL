import os
import importlib
import pandas as pd

# 变体配置：(模块名, 输出文件后缀, 描述)
variants = [
    ('modelB_full', '_full', '完整模型'),
    ('modelB_no_cl', '_no_cl', '无对比学习'),
    ('modelB_no_gate', '_no_gate', '无门控融合'),
    ('modelB_no_decouple', '_no_decouple', '无多模态解耦')
]

# 真值文件路径（请根据您的实际文件名修改）
truth_files = {
    'Dream': 'KOL_Rank_Dream_Video.csv',
    'Hair': 'KOL_Rank_PinkHair_Video.csv'
}

# 数据文件路径
data_files = {
    'Dream': ('final_dream_nodes.csv', 'final_dream_edges.csv'),
    'Hair': ('final_hair_nodes.csv', 'final_hair_edges.csv')
}

# 记录每个变体生成的输出文件
output_records = []

for module_name, suffix, desc in variants:
    # 动态导入模型模块
    model_module = importlib.import_module(module_name)
    run_func = getattr(model_module, 'run_final_optimized_analysis')

    for topic in ['Dream', 'Hair']:
        nodes_path, edges_path = data_files[topic]
        print(f"\n========== Running {desc} on {topic} ==========")

        # 注意：这里不传递 ablation_mode，因为每个变体文件内部已经固定了消融逻辑
        run_func(nodes_path, edges_path, topic, scale_factor=0.5)

        # 模型默认输出文件名（与 modelB.py 中一致）
        default_name = f'KOL_GNN_Rank_{topic}_Final_Optimized.csv'
        # 新的带后缀的文件名
        new_name = f'KOL_GNN_Rank_{topic}{suffix}_Final_Optimized.csv'

        if os.path.exists(default_name):
            # 如果新文件名已存在，先删除（避免冲突）
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(default_name, new_name)
            print(f"   结果已保存到 {new_name}")
        else:
            print(f"   ⚠️ 未找到默认输出文件 {default_name}，跳过")
            continue

        output_records.append({
            'topic': topic,
            'desc': desc,
            'out_csv': new_name,
            'truth_csv': truth_files[topic]
        })

# 保存文件列表供后续细粒度评估使用
df_records = pd.DataFrame(output_records)
df_records.to_csv('ablation_file_list.csv', index=False)
print("\n各变体输出文件列表已保存到 ablation_file_list.csv")