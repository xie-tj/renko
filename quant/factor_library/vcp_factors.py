"""
VCP形态因子库 - Volatility Contraction Pattern

核心因子：
1. 波动率收敛 (VCP) - 动态锚定法
2. 枢轴邻近/突破 (Pivot)
3. 均线结构 - 多头排列
4. 均线斜率
5. 收盘强度
6. 低位反包信号
7. 派发日惩罚
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VCPFactors:
    """VCP形态因子 - 马克·米勒维尼体系"""
    
    @staticmethod
    def volatility_contraction(df: pd.DataFrame, short_window: int = 5, long_window: int = 35) -> pd.Series:
        """
        波动率收敛 (VCP - 动态锚定法)
        
        逻辑：近期价格波动率(5日)显著低于整体波动率(35日)
        仅考量价格振幅收敛，不强制要求成交量萎缩
        
        Returns:
            VCP比率：越小表示收敛越好（0.5以下为佳）
        """
        df = df.copy()
        
        # 计算日收益率
        df['return'] = df['close'].pct_change()
        
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            short_std = df['return'].rolling(short_window, min_periods=1).std()
            long_std = df['return'].rolling(long_window, min_periods=1).std()
        else:
            short_std = df.groupby('code')['return'].transform(
                lambda x: x.rolling(short_window, min_periods=1).std()
            )
            long_std = df.groupby('code')['return'].transform(
                lambda x: x.rolling(long_window, min_periods=1).std()
            )
        
        vcp_ratio = short_std / long_std.replace(0, np.nan)
        vcp_ratio = vcp_ratio.clip(upper=2.0)
        
        return vcp_ratio.fillna(1.0)
    
    @staticmethod
    def pivot_proximity(df: pd.DataFrame, window: int = 35) -> pd.Series:
        """
        枢轴邻近/突破 (5日均值法)
        
        逻辑：(区间最高+区间最低+当前收盘)/3 作为枢轴
        区间极值计算截止到昨天（避免未来数据）
        
        Returns:
            价格相对枢轴的位置 (>1表示在枢轴上方)
        """
        df = df.copy()
        
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            high = df['high'].shift(1).rolling(window, min_periods=1).max()
            low = df['low'].shift(1).rolling(window, min_periods=1).min()
        else:
            high = df.groupby('code')['high'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )
            low = df.groupby('code')['low'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
        
        pivot = (high + low + df['close']) / 3
        proximity = df['close'] / pivot.replace(0, np.nan)
        
        return proximity.fillna(1.0)
    
    @staticmethod
    def ma_alignment(df: pd.DataFrame) -> pd.Series:
        """
        均线结构 - MA20 > MA28 > MA57 三均线多头排列 (铁律因子)
        
        逻辑：基于成功案例分析，MA20 > MA28 > MA57 是100%满足的铁律
        这是一个斐波那契周期的三均线组合，代表中长期趋势确认
        
        Returns:
            三均线多头排列得分 (1=完全多头排列, 0=不满足, 0.5=部分满足)
        """
        df = df.copy()
        
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            ma20 = df['close'].rolling(20, min_periods=1).mean()
            ma28 = df['close'].rolling(28, min_periods=1).mean()
            ma57 = df['close'].rolling(57, min_periods=1).mean()
        else:
            ma20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
            ma28 = df.groupby('code')['close'].transform(lambda x: x.rolling(28, min_periods=1).mean())
            ma57 = df.groupby('code')['close'].transform(lambda x: x.rolling(57, min_periods=1).mean())
        
        # 计算三均线多头得分
        # MA20 > MA28 得 0.5 分
        # MA28 > MA57 得 0.5 分
        score = (ma20 > ma28).astype(float) * 0.5 + (ma28 > ma57).astype(float) * 0.5
        
        return score
    
    @staticmethod
    def ma_slope(df: pd.DataFrame, ma_window: int = 5, slope_window: int = 5) -> pd.Series:
        """
        均线斜率 - Trend Slope
        
        逻辑：均线不仅要多头排列，运行方向必须是朝上的
        
        Returns:
            均线斜率（正数表示向上，负数表示向下）
        """
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            ma = df['close'].rolling(ma_window, min_periods=1).mean()
            slope = (ma - ma.shift(slope_window)) / ma.shift(slope_window)
        else:
            ma = df.groupby('code')['close'].transform(lambda x: x.rolling(ma_window, min_periods=1).mean())
            slope = (ma - ma.shift(slope_window)) / ma.shift(slope_window)
        
        return slope.fillna(0)
    
    @staticmethod
    def closing_range(df: pd.DataFrame) -> pd.Series:
        """
        收盘强度 - Closing Range
        
        逻辑：衡量收盘价在今天全天震荡区间中的相对位置
        反映资金做多过夜的意愿
        
        Returns:
            0-1之间，越接近1表示收盘越强（接近最高价）
        """
        range_size = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / range_size.replace(0, np.nan)
        return close_position.fillna(0.5)
    
    @staticmethod
    def bullish_reversal(df: pd.DataFrame) -> pd.Series:
        """
        低位反包信号 - Bullish Reversal / Engulfing
        
        逻辑：典型的阳线吞没阴线形态，代表日内或隔日情绪反转
        条件：
        1. 昨日阴线（收盘 < 开盘）
        2. 今日阳线（收盘 > 开盘）
        3. 今日实体完全吞没昨日实体
        
        Returns:
            反包强度（0-1，0表示无反包）
        """
        df = df.copy()
        
        df['prev_open'] = df['open'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        df['today_body'] = df['close'] - df['open']
        df['prev_body'] = df['prev_close'] - df['prev_open']
        
        df['prev_bear'] = df['prev_body'] < 0
        df['today_bull'] = df['today_body'] > 0
        
        df['engulf_low'] = df['low'] <= df[['open', 'close']].shift(1).min(axis=1)
        df['engulf_high'] = df['high'] >= df[['open', 'close']].shift(1).max(axis=1)
        
        df['reversal_strength'] = df['today_body'] / df['prev_body'].abs().replace(0, np.nan)
        df['reversal_strength'] = df['reversal_strength'].fillna(0)
        
        is_reversal = df['prev_bear'] & df['today_bull'] & df['engulf_low'] & df['engulf_high']
        
        return (is_reversal.astype(float) * df['reversal_strength'].clip(0, 1))
    
    @staticmethod
    def distribution_penalty(df: pd.DataFrame, volume_window: int = 20) -> pd.Series:
        """
        派发日惩罚 - Distribution Day
        
        逻辑：监控"放量大阴线"，主力资金出货的典型特征
        条件：
        1. 阴线（收盘 < 开盘）- 且不是假阴真阳（收盘 > 开盘）
        2. 振幅 > 5%
        3. 成交量放大（> 均量）
        
        注意：假阴真阳（收盘 > 开盘但K线显示为阴线）不算派发
        
        Returns:
            惩罚值（0-1，越大表示派发信号越强）
        """
        df = df.copy()
        
        # 真阴线：收盘 < 开盘（排除假阴真阳）
        is_true_bearish = df['close'] < df['open']
        
        # 假阴真阳：虽然K线显示阴线（收盘 < 昨日收盘），但实际收盘 > 开盘
        # 这种情况不算派发，需要排除
        is_fake_bearish = (df['close'] > df['open']) & (df['close'] < df['close'].shift(1))
        
        # 真正的派发日：真阴线且不是假阴真阳
        is_distribution_day = is_true_bearish & ~is_fake_bearish
        
        amplitude = (df['high'] - df['low']) / df['open']
        large_range = amplitude > 0.05
        
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            avg_volume = df['volume'].rolling(volume_window, min_periods=1).mean()
        else:
            avg_volume = df.groupby('code')['volume'].transform(
                lambda x: x.rolling(volume_window, min_periods=1).mean()
            )
        volume_spike = df['volume'] > avg_volume * 1.2
        
        # 同时满足三个条件才是真派发
        is_distribution = is_distribution_day & large_range & volume_spike
        
        drop_pct = (df['open'] - df['close']) / df['open']
        volume_ratio = df['volume'] / avg_volume.replace(0, np.nan)
        
        penalty = drop_pct * volume_ratio.fillna(1)
        
        return (is_distribution.astype(float) * penalty.clip(0, 1))
    @staticmethod
    def upper_shadow_penalty(df: pd.DataFrame) -> pd.Series:
        """
        上影线惩罚因子（P惩罚项）
        
        逻辑：上影线越长，表示上方抛压越大，应该惩罚
        计算：
        - 上影线长度 = 最高价 - max(开盘价, 收盘价)
        - 上影线占比 = 上影线长度 / 总振幅
        - 当日最大涨幅 = (最高价 - 开盘价) / 开盘价
        - 因子值 = (1 - 上影线占比) × (当日最大涨幅)^2
        
        效果：涨幅越大但上影线越长，惩罚越重（平方放大效果）
        
        Returns:
            因子值（越大越好，越小表示上影线问题越严重）
        """
        df = df.copy()
        
        # 上影线长度（从实体顶端到最高价）
        df['实体顶端'] = df[['open', 'close']].max(axis=1)
        df['上影线长度'] = df['high'] - df['实体顶端']
        
        # 总振幅
        df['振幅'] = df['high'] - df['low']
        df['振幅'] = df['振幅'].replace(0, np.nan)
        
        # 上影线占比（限制在0-1之间）
        df['上影线占比'] = (df['上影线长度'] / df['振幅']).fillna(0).clip(0, 1)
        
        # 当日最大涨幅（从开盘到最高）
        df['当日最大涨幅'] = (df['high'] - df['open']) / df['open'].replace(0, np.nan)
        df['当日最大涨幅'] = df['当日最大涨幅'].fillna(0)
        
        # 惩罚：上影线越长，因子值越小；同时考虑当日最大涨幅的平方
        # 无上影线(光头) = 1 × 涨幅^2, 全是上影线 = 0
        df['上影线因子'] = (1 - df['上影线占比']) * (df['当日最大涨幅'] ** 2)
        
        return df['上影线因子']
    @staticmethod
    def closing_range_5d(df: pd.DataFrame) -> pd.Series:
        """
        5日突破强度 - 5-Day Breakout Strength
        
        逻辑：当日收盘价相对于前4天收盘价区间的突破强度
        计算：
        - 前4天(不含今日)的收盘价区间 [min, max]
        - 今日收盘价在该区间中的百分位
        - 突破前4天最高价 > 100%
        - 跌破前4天最低价 < 0%
        - 在区间内 0% ~ 100%
        
        Returns:
            百分位值，可正可负，>100%表示突破前高
        """
        # 计算前4天的收盘价区间（不包含今天）
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            high_4d = df['close'].shift(1).rolling(4, min_periods=1).max()
            low_4d = df['close'].shift(1).rolling(4, min_periods=1).min()
        else:
            high_4d = df.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
            low_4d = df.groupby('code')['close'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).min())
        
        # 计算今日收盘价在前4天区间中的位置
        # 公式：(close - low_4d) / (high_4d - low_4d) * 100
        range_4d = high_4d - low_4d
        position = (df['close'] - low_4d) / range_4d.replace(0, np.nan) * 100
        
        return position.fillna(50)  # 默认50%（中间位置）
    
    @staticmethod
    def volatility_5day(df: pd.DataFrame) -> pd.Series:
        """
        前5日波动率（基于开盘价和收盘价）
        
        逻辑：计算前5日的价格波动程度
        使用 (high - low) / open 来衡量每日波动
        然后取5日平均
        
        Returns:
            前5日波动率（越小表示前期波动越小，越有利于突破）
        """
        df = df.copy()
        
        # 计算每日波动率
        df['daily_vol'] = (df['high'] - df['low']) / df['open']
        
        # 前5日平均波动率（不含当日）
        if 'code' not in df.columns or df['code'].nunique() <= 1:
            vol_5d = df['daily_vol'].shift(1).rolling(5, min_periods=1).mean()
        else:
            vol_5d = df.groupby('code')['daily_vol'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
        
        return vol_5d.fillna(0)
    
