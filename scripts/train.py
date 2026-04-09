#!/usr/bin/env python3
"""
简化版训练脚本 - 可以直接运行 python main.py train
"""
import sys
from pathlib import Path

# 添加项目根目录到路径（train.py 现在在 scripts/ 子目录中）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

print("=" * 80)
print(" 训练因子权重 - 正负样本联合训练")
print("=" * 80)

# 设置路径
data_dir = project_root / "data"

# 加载信号池
print("\n加载信号池...")
signal_file = data_dir / "processed" / "signal_pool.csv"
if not signal_file.exists():
    print("错误: 信号池不存在")
    sys.exit(1)

df = pd.read_csv(signal_file)
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2020-01-01']

print(f"加载数据: {len(df)} 条信号")

# 计算未来3日收益
print("计算历史收益...")
df = df.sort_values(['code', 'date'])
df['future_3d_return'] = df.groupby('code')['close'].shift(-3) / df.groupby('code')['close'].shift(-1) - 1
df = df[df['future_3d_return'].notna()]

# 划分正负样本（收益>=3%为正）
df['is_positive'] = df['future_3d_return'] >= 0.03
pos_df = df[df['is_positive'] == True]
neg_df = df[df['is_positive'] == False]

print(f"正样本: {len(pos_df)} ({len(pos_df)/len(df)*100:.1f}%)")
print(f"负样本: {len(neg_df)} ({len(neg_df)/len(df)*100:.1f}%)")

# 7个因子（更新后的列表）
factors = ['renko_strength_entity', 'vcp_ratio', 'pivot_proximity',
           'closing_range', 'ma_alignment',
           'upper_shadow_penalty', 'volatility_5day']

# 检查因子是否存在
available_factors = [f for f in factors if f in df.columns]
if len(available_factors) < len(factors):
    missing = set(factors) - set(available_factors)
    print(f"\n警告: 缺少因子 {missing}")
    print(f"将使用可用因子: {available_factors}")
    factors = available_factors

# 使用全部样本训练（不再采样）
print(f"\n使用全部样本训练: {len(df)} 条")
train_df = df.copy()

# K-Means聚类
print("\nK-Means聚类...")
factor_data = train_df[factors].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
train_df['cluster'] = kmeans.fit_predict(factor_data)

print("聚类分布:")
for c in range(3):
    cluster_df = train_df[train_df['cluster'] == c]
    pos_count = cluster_df['is_positive'].sum()
    neg_count = len(cluster_df) - pos_count
    print(f"  聚类 {c}: {len(cluster_df)}条 (正{pos_count}/负{neg_count})")

# 为每个聚类计算权重
print("\n" + "=" * 80)
print(" 计算权重")
print("=" * 80)

cluster_weights = {}

for c in range(3):
    cluster_df = train_df[train_df['cluster'] == c]
    pos_count = cluster_df['is_positive'].sum()
    neg_count = len(cluster_df) - pos_count

    print(f"\n聚类 {c} ({len(cluster_df)}条, 正{pos_count}/负{neg_count}):")

    # 计算IC
    ics = {}
    for f in factors:
        ic = cluster_df[f].corr(cluster_df['is_positive'])
        ics[f] = ic

    # 基于IC绝对值分配权重，保留方向
    abs_ics = np.array([abs(ics[f]) for f in factors])
    if abs_ics.sum() > 0:
        weights_arr = abs_ics / abs_ics.sum()
    else:
        weights_arr = np.ones(len(factors)) / len(factors)

    cluster_weights[str(c)] = {}
    for i, f in enumerate(factors):
        w = weights_arr[i]
        direction = 1 if ics[f] >= 0 else -1

        # 计算该聚类中该因子的适宜区间（正样本的25%-75%分位数）
        pos_values = cluster_df[cluster_df['is_positive'] == True][f].dropna()
        if len(pos_values) > 10:
            optimal_low = pos_values.quantile(0.25)
            optimal_high = pos_values.quantile(0.75)
            optimal_mean = pos_values.mean()
        else:
            optimal_low = cluster_df[f].quantile(0.25)
            optimal_high = cluster_df[f].quantile(0.75)
            optimal_mean = cluster_df[f].mean()

        cluster_weights[str(c)][f] = {
            'weight': float(w),
            'ic': float(ics[f]),
            'direction': direction,
            'optimal_range': [float(optimal_low), float(optimal_high)],
            'optimal_mean': float(optimal_mean)
        }

    # 显示
    for f in factors[:3]:
        w = cluster_weights[str(c)][f]['weight']
        ic = cluster_weights[str(c)][f]['ic']
        d = '+' if cluster_weights[str(c)][f]['direction'] > 0 else '-'
        opt_range = cluster_weights[str(c)][f]['optimal_range']
        print(f"  {f:20s}: W={w:.3f}, IC={ic:+.3f}, {d}, 适宜区间=[{opt_range[0]:.2f}, {opt_range[1]:.2f}]")

# 保存权重
output_file = data_dir / "processed" / "factor_weights.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cluster_weights, f, ensure_ascii=False, indent=2)

print(f"\n权重已保存: {output_file}")

# 验证效果
print("\n" + "=" * 80)
print(" 验证效果")
print("=" * 80)

# 计算得分
train_df['score'] = 0.0
for idx, row in train_df.iterrows():
    c = str(int(row['cluster']))
    if c in cluster_weights:
        score = 0
        for f in factors:
            if f in row and pd.notna(row[f]):
                w = cluster_weights[c][f]['weight']
                direction = cluster_weights[c][f]['direction']
                score += row[f] * w * direction * 100
        train_df.at[idx, 'score'] = score

# 计算AUC
try:
    auc = roc_auc_score(train_df['is_positive'], train_df['score'])
    print(f"\nAUC: {auc:.4f}")
except:
    print(f"\nAUC: 计算失败")

# 计算正负样本得分差异
pos_scores = train_df[train_df['is_positive'] == True]['score']
neg_scores = train_df[train_df['is_positive'] == False]['score']

print(f"正样本平均得分: {pos_scores.mean():.2f}")
print(f"负样本平均得分: {neg_scores.mean():.2f}")
print(f"区分度: {pos_scores.mean() - neg_scores.mean():.2f}")

print("\n" + "=" * 80)
print(" 训练完成!")
print("=" * 80)
