#!/usr/bin/env python3
"""
砖型图量化交易系统 - 主程序
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


class RenkoQuantSystem:
    """砖型图量化交易系统"""

    def __init__(self):
        # 使用相对路径，避免硬编码绝对路径
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"

    def build(self):
        """
        构建信号池 - 计算所有因子并进行标准化
        """
        print("=" * 80)
        print(" 构建信号池")
        print("=" * 80)

        from quant.calc_factors import FactorCalculator
        from utils.csv_manager import CSVManager

        # 初始化
        csv = CSVManager(self.data_dir)
        stocks = csv.list_all_stocks()

        print(f"\n发现 {len(stocks)} 只股票")

        all_signals = []

        for i, code in enumerate(stocks, 1):
            # 显示进度
            progress = int((i / len(stocks)) * 30)
            bar = '█' * progress + '░' * (30 - progress)
            print(f"\r\033[K  [{bar}] {i}/{len(stocks)} {code}", end='', flush=True)

            # 读取股票数据
            df = csv.read_stock(code)
            if df.empty:
                continue

            # 转为正序
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True).reset_index(drop=True)

            # 先计算砖型图相关列（用于信号判断）
            df = self._calc_renko_signals(df)

            # 提取信号（砖型图信号为True）
            if '砖型图信号' in df.columns:
                signals = df[df['砖型图信号'] == True].copy()
                if not signals.empty:
                    # 计算因子（只在信号行计算，提高效率）
                    signals = FactorCalculator.calc_all_factors(signals, code=code, show_progress=False)
                    signals['code'] = str(code)  # 确保代码为字符串格式
                    # 添加股票名称
                    name = csv.get_stock_name(code)
                    if name:
                        signals['name'] = name
                    all_signals.append(signals)

            # 保存包含因子的完整数据（可选）
            df_with_factors = FactorCalculator.calc_all_factors(df, code=code, show_progress=False)
            csv.save_stock(code, df_with_factors)

        print("\n")

        # 合并所有信号
        if all_signals:
            # 检查并修复每个信号的列名重复问题
            for i, s in enumerate(all_signals):
                # 如果有重复列名，只保留第一个
                if not s.columns.is_unique:
                    s = s.loc[:, ~s.columns.duplicated()]
                    all_signals[i] = s

            # 统一列名：只保留所有信号共有的列
            common_cols = set(all_signals[0].columns)
            for s in all_signals[1:]:
                common_cols &= set(s.columns)
            common_cols = sorted(list(common_cols))  # 排序确保顺序一致

            print(f"\n共有列数: {len(common_cols)}")

            # 只保留共有列，并确保列名唯一
            unified_signals = []
            for s in all_signals:
                # 选择共有列
                s_subset = s[common_cols].copy()
                # 再次检查列名唯一性
                if not s_subset.columns.is_unique:
                    s_subset = s_subset.loc[:, ~s_subset.columns.duplicated()]
                unified_signals.append(s_subset)

            # 使用 join='inner' 确保只使用共有列
            signal_pool = pd.concat(unified_signals, ignore_index=True, join='inner')
            print(f"信号池: {len(signal_pool)} 条信号")

            # 标准化因子
            print("\n标准化因子...")
            factors = ['renko_strength_entity', 'vcp_ratio', 'pivot_proximity',
                      'closing_range', 'ma_alignment',
                      'upper_shadow_penalty', 'volatility_5day']

            for factor in factors:
                if factor in signal_pool.columns:
                    # 去极值 (1%分位数)
                    lower = signal_pool[factor].quantile(0.01)
                    upper = signal_pool[factor].quantile(0.99)
                    signal_pool[factor] = signal_pool[factor].clip(lower, upper)

                    # 标准化 (Z-Score)
                    mean = signal_pool[factor].mean()
                    std = signal_pool[factor].std()
                    if std > 0:
                        signal_pool[factor] = (signal_pool[factor] - mean) / std

            # 保存信号池
            output_file = self.data_dir / "processed" / "signal_pool.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            signal_pool.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"信号池已保存: {output_file}")

            # 验证标准化
            print("\n验证标准化...")
            self._verify_standardization(signal_pool, factors)
        else:
            print("警告: 没有生成任何信号")

    def _calc_renko_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算砖型图信号（今日红柱等）
        简化版，只计算信号判断所需的列
        """
        df = df.copy()

        # 计算砖型图
        df['HHV4'] = df['high'].rolling(window=4, min_periods=1).max()
        df['LLV4'] = df['low'].rolling(window=4, min_periods=1).min()

        df['VAR1A'] = (df['HHV4'] - df['close']) / (df['HHV4'] - df['LLV4']).replace(0, np.nan) * 100 - 90
        df['VAR1A'] = df['VAR1A'].replace([np.inf, -np.inf], 0).fillna(0)

        # SMA平滑
        def sma(series, n, m=1):
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                if pd.isna(series.iloc[i]):
                    result.iloc[i] = result.iloc[i-1]
                else:
                    result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
            return result

        df['VAR2A'] = sma(df['VAR1A'], 4, 1) + 100

        df['VAR3A'] = (df['close'] - df['LLV4']) / (df['HHV4'] - df['LLV4']).replace(0, np.nan) * 100
        df['VAR3A'] = df['VAR3A'].replace([np.inf, -np.inf], 0).fillna(0)
        df['VAR4A'] = sma(df['VAR3A'], 6, 1)
        df['VAR5A'] = sma(df['VAR4A'], 6, 1) + 100

        df['VAR6A'] = df['VAR5A'] - df['VAR2A']
        df['砖型图'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)

        # 红柱绿柱
        df['砖型图_前1'] = df['砖型图'].shift(1)
        df['砖型图_前2'] = df['砖型图'].shift(2)

        # AA:=REF(砖型图,1)<砖型图  {今天涨，出红柱}
        df['AA'] = df['砖型图_前1'] < df['砖型图']

        # BB:=REF(砖型图,1)>砖型图  {今天跌，出绿柱}
        df['BB'] = df['砖型图_前1'] > df['砖型图']

        # 昨日绿柱:=REF(砖型图,2)-REF(砖型图,1)
        df['昨日绿柱'] = df['砖型图_前2'] - df['砖型图_前1']
        df['昨日绿柱'] = df['昨日绿柱'].where(df['砖型图_前2'] > df['砖型图_前1'], 0)

        # 今日红柱:=砖型图-REF(砖型图,1)
        df['今日红柱'] = df['砖型图'] - df['砖型图_前1']
        df['今日红柱'] = df['今日红柱'].where(df['砖型图'] > df['砖型图_前1'], 0)

        # 力度达标:=今日红柱 >= (昨日绿柱 * 2 / 3)
        df['力度达标'] = df['今日红柱'] >= (df['昨日绿柱'] * 2 / 3)

        # CC:=REF(BB,1) AND AA  {昨天是绿柱，今天是红柱}
        df['BB_前1'] = df['BB'].shift(1)
        df['CC'] = df['BB_前1'] & df['AA']

        # 知行趋势线计算
        df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['知行短期趋势线'] = df['EMA10'].ewm(span=10, adjust=False).mean()

        df['MA14'] = df['close'].rolling(window=14, min_periods=1).mean()
        df['MA28'] = df['close'].rolling(window=28, min_periods=1).mean()
        df['MA57'] = df['close'].rolling(window=57, min_periods=1).mean()
        df['MA114'] = df['close'].rolling(window=114, min_periods=1).mean()
        df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4

        # 条件1:=知行短期趋势线 > 知行多空线
        df['条件1'] = df['知行短期趋势线'] > df['知行多空线']

        # 条件2:=CLOSE > 知行多空线
        df['条件2'] = df['close'] > df['知行多空线']

        # 最终信号: CC AND 力度达标 AND 条件1 AND 条件2 AND 今日红柱 >= 5
        df['砖型图信号'] = df['CC'] & df['力度达标'] & df['条件1'] & df['条件2'] & (df['今日红柱'] >= 5)

        return df

    def _verify_standardization(self, df, factors):
        """验证因子标准化状态"""
        print("\n" + "-" * 60)
        print(" 标准化验证结果")
        print("-" * 60)

        all_ok = True
        for factor in factors:
            if factor not in df.columns:
                print(f"  [!] {factor}: 因子不存在")
                all_ok = False
                continue

            values = df[factor].dropna()
            mean = values.mean()
            std = values.std()

            is_ok = abs(mean) < 0.1 and abs(std - 1.0) < 0.1
            status = "[OK]" if is_ok else "[X]"

            print(f"  {factor:25s}: 均值={mean:+.4f}, 标准差={std:.4f} {status}")

            if not is_ok:
                all_ok = False

        print("-" * 60)
        if all_ok:
            print("  [OK] 所有因子已成功标准化")
        else:
            print("  [X] 部分因子未标准化")
        print("-" * 60)

    def train(self, use_latest_only: bool = False, recent_days: int = 365):
        """
        训练因子权重 - 正负样本联合训练

        Args:
            use_latest_only: 是否只使用最近数据（加速训练）
            recent_days: 使用最近多少天的数据（默认365天）
        """
        """
        训练因子权重 - 正负样本联合训练

        核心逻辑：
        1. 使用正负样本一起训练（收益>=3%为正样本）
        2. 使用K-Means聚类（3个聚类）
        3. 正确处理IC方向（IC为负时反向使用因子）
        4. 优化目标是最大化正负样本区分度
        """
        print("=" * 80)
        print(" 训练因子权重 - 正负样本联合训练")
        print("=" * 80)

        # 加载信号池
        print("\n加载信号池...")
        signal_file = self.data_dir / "processed" / "signal_pool.csv"
        if not signal_file.exists():
            print("错误: 信号池不存在")
            return

        df = pd.read_csv(signal_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # 选择数据范围
        if use_latest_only:
            # 只使用最近N天的数据
            latest_date = df['date'].max()
            cutoff_date = latest_date - pd.Timedelta(days=recent_days)
            df = df[df['date'] >= cutoff_date]
            print(f"使用最近{recent_days}天数据: {len(df)} 条信号")
        else:
            df = df[df['date'] >= '2020-01-01']
            print(f"加载全部数据: {len(df)} 条信号")

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

        # 7个因子（train方法，更新后的列表）
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

            # 计算IC（与is_positive的相关系数）
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

            # 显示全部因子
            for f in factors:
                w = cluster_weights[str(c)][f]['weight']
                ic = cluster_weights[str(c)][f]['ic']
                d = '+' if cluster_weights[str(c)][f]['direction'] > 0 else '-'
                opt_range = cluster_weights[str(c)][f]['optimal_range']
                print(f"  {f:25s}: W={w:.3f}, IC={ic:+.3f}, {d}, 适宜区间=[{opt_range[0]:.2f}, {opt_range[1]:.2f}]")

        # 保存权重
        output_file = self.data_dir / "processed" / "factor_weights.json"
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
            auc = 0.5
            print(f"\nAUC: 计算失败")

        # 计算正负样本得分差异
        pos_scores = train_df[train_df['is_positive'] == True]['score']
        neg_scores = train_df[train_df['is_positive'] == False]['score']

        pos_mean = pos_scores.mean()
        neg_mean = neg_scores.mean()
        discrimination = pos_mean - neg_mean

        print(f"正样本平均得分: {pos_mean:.2f}")
        print(f"负样本平均得分: {neg_mean:.2f}")
        print(f"区分度: {discrimination:.2f}")

        # 保存评估指标到权重文件
        cluster_weights['_evaluation'] = {
            'auc': float(auc),
            'pos_mean_score': float(pos_mean),
            'neg_mean_score': float(neg_mean),
            'discrimination': float(discrimination),
            'train_samples': len(train_df),
            'positive_samples': int(train_df['is_positive'].sum()),
            'negative_samples': int((~train_df['is_positive']).sum())
        }

        # 重新保存权重（包含评估指标）
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_weights, f, ensure_ascii=False, indent=2)

        print(f"\n评估指标已保存到权重文件")

        print("\n" + "=" * 80)
        print(" 训练完成!")
        print("=" * 80)

    def select(self, top_n: int = 20):
        """
        执行选股

        Args:
            top_n: 返回前N只股票
        """
        print("=" * 80)
        print(" 执行选股")
        print("=" * 80)

        # 加载信号池
        signal_file = self.data_dir / "processed" / "signal_pool.csv"
        if not signal_file.exists():
            print("错误: 信号池不存在，请先执行: python main.py build")
            return

        df = pd.read_csv(signal_file)
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()
        latest_signals = df[df['date'] == latest_date].copy()

        print(f"最新日期: {latest_date.strftime('%Y-%m-%d')}")
        print(f"信号数量: {len(latest_signals)} 只")

        # 检查列是否存在
        print(f"\n可用列: {list(latest_signals.columns)}")

        # 过滤ST/退市（如果有name列）
        if 'name' in latest_signals.columns:
            latest_signals = latest_signals[~latest_signals['name'].str.contains('ST|退', na=False)]
            print(f"过滤ST后: {len(latest_signals)} 只")
        else:
            print("警告: 没有name列，跳过ST过滤")

        # 7个因子（select方法，更新后的列表）
        factors = ['renko_strength_entity', 'vcp_ratio', 'pivot_proximity',
                   'closing_range', 'ma_alignment',
                   'upper_shadow_penalty', 'volatility_5day']

        # 检查可用因子
        available_factors = [f for f in factors if f in latest_signals.columns]
        factors = available_factors

        # 加载权重
        weights_file = self.data_dir / "processed" / "factor_weights.json"
        if weights_file.exists():
            with open(weights_file, 'r', encoding='utf-8') as f:
                weights = json.load(f)
            print("已加载权重")
        else:
            print("警告: 权重文件不存在，使用默认权重")
            weights = {}

        # K-Means聚类
        print("\nK-Means聚类...")
        factor_data = latest_signals[factors].fillna(0)
        
        # 根据信号数量动态调整聚类数
        n_samples = len(latest_signals)
        n_clusters = min(3, n_samples)  # 聚类数不超过样本数
        
        if n_clusters < 3:
            print(f"  警告: 信号数量({n_samples})少于默认聚类数(3)，调整为{n_clusters}个聚类")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(factor_data)
        latest_signals['cluster'] = clusters

        print(f"聚类分布: " + ", ".join([f"聚类{i}={sum(clusters==i)}" for i in range(n_clusters)]))

        # 计算得分
        scores = []
        for _, row in latest_signals.iterrows():
            cluster = str(int(row['cluster']))

            if cluster in weights:
                score = 0
                for f in factors:
                    if f in row and pd.notna(row[f]):
                        w = weights[cluster][f]['weight']
                        direction = weights[cluster][f].get('direction', 1)
                        score += row[f] * w * direction * 100
            else:
                valid_factors = [row[f] for f in factors if f in row and pd.notna(row[f])]
                score = sum(valid_factors) / len(valid_factors) if valid_factors else 0

            scores.append(score)

        latest_signals['score'] = scores
        latest_signals = latest_signals.sort_values('score', ascending=False)

        # 显示结果
        print(f"\n得分统计:")
        print(f"  最高分: {latest_signals['score'].max():.2f}")
        print(f"  最低分: {latest_signals['score'].min():.2f}")
        print(f"  平均分: {latest_signals['score'].mean():.2f}")

        print(f"\n推荐股票 (Top {top_n}):")
        print("-" * 90)
        print(f"{'排名':<6}{'代码':<10}{'名称':<12}{'得分':<10}{'聚类':<8}{'收盘价':<10}")
        print("-" * 90)

        for i, (_, row) in enumerate(latest_signals.head(top_n).iterrows(), 1):
            code = str(int(row['code'])).zfill(6) if pd.notna(row['code']) else '000000'
            name = str(row.get('name', '-'))[:10] if pd.notna(row.get('name')) else '-'
            print(f"{i:<6}{code:<10}{name:<12}"
                  f"{row['score']:<10.2f}{int(row['cluster']):<8}{row['close']:<10.2f}")

        # 保存结果
        output_file = self.data_dir / "output" / f"selection_{latest_date.strftime('%Y%m%d')}.csv"
        output_file.parent.mkdir(exist_ok=True)

        # 只保存存在的列
        save_cols = ['date', 'code', 'score', 'cluster', 'close']
        if 'name' in latest_signals.columns:
            save_cols.insert(2, 'name')
        if 'signal_type' in latest_signals.columns:
            save_cols.insert(3, 'signal_type')

        output_df = latest_signals.head(top_n)[save_cols].copy()
        output_df['code'] = output_df['code'].apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else '000000')
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n选股结果已保存: {output_file}")

        return latest_signals.head(top_n)

    def update(self, max_stocks: int = None):
        """
        更新股票数据 - 从Tushare获取最新数据

        Args:
            max_stocks: 最大更新股票数量，None表示全部
        """
        print("=" * 80)
        print(" 更新股票数据")
        print("=" * 80)

        from utils.akshare_fetcher import TushareFetcher
        from utils.csv_manager import get_csv_manager

        fetcher = TushareFetcher(data_dir=str(self.data_dir))
        csv_manager = get_csv_manager(str(self.data_dir))

        # 检查是否已有数据
        existing_stocks = csv_manager.list_all_stocks()
        
        if not existing_stocks:
            print("\n没有现有数据，执行首次全量抓取...")
            print("=" * 80)
            fetcher.init_full_data(max_stocks=max_stocks, delay=0.5)
        else:
            # 使用 daily_update 方法更新到今天
            print(f"\n发现 {len(existing_stocks)} 只现有股票，执行增量更新...")
            fetcher.daily_update(max_stocks=max_stocks)

        print("\n更新完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='砖型图量化交易系统')
    parser.add_argument('command', choices=['update', 'build', 'train', 'select', 'run'], help='命令: update-更新数据, build-构建信号池, train-训练权重, select-执行选股, run-一键运行完整流程')
    parser.add_argument('--top', type=int, default=20, help='选股数量，默认20')
    parser.add_argument('--max-stocks', type=int, default=None, help='最大更新股票数量')
    parser.add_argument('--recent-days', type=int, default=365, help='训练时使用最近多少天的数据，默认365天')
    parser.add_argument('--fast-train', action='store_true', help='快速训练模式（只使用最近数据）')
    parser.add_argument('--skip-update', action='store_true', help='跳过数据更新')
    parser.add_argument('--skip-build', action='store_true', help='跳过构建信号池')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练权重')

    args = parser.parse_args()

    system = RenkoQuantSystem()

    if args.command == 'update':
        system.update(max_stocks=args.max_stocks)
    elif args.command == 'build':
        system.build()
    elif args.command == 'train':
        system.train(use_latest_only=args.fast_train, recent_days=args.recent_days)
    elif args.command == 'select':
        system.select(top_n=args.top)
    elif args.command == 'run':
        # 一键运行完整流程
        print("\n" + "=" * 80)
        print(" 一键运行完整流程")
        print("=" * 80)

        if not args.skip_update:
            system.update(max_stocks=args.max_stocks)
        else:
            print("\n跳过数据更新")

        if not args.skip_build:
            system.build()
        else:
            print("\n跳过构建信号池")

        if not args.skip_train:
            system.train(use_latest_only=args.fast_train, recent_days=args.recent_days)
        else:
            print("\n跳过训练权重")

        system.select(top_n=args.top)
