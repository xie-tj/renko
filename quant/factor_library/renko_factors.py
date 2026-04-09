#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
砖型图专用因子 - 涨幅5%阈值版

核心逻辑：
1. 支持两种形态：反包 + 突破
2. 上影线阈值：30%
3. 涨幅<5%不惩罚（即使上影线长）
4. 涨幅>=5%且有上影线才惩罚
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信SMA平滑移动平均"""
    result = pd.Series(index=series.index, dtype=float)
    result.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            result.iloc[i] = result.iloc[i-1]
        else:
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
    
    return result


class RenkoFactors:
    """砖型图相关因子"""

    @staticmethod
    def renko_strength_entity(df: pd.DataFrame) -> pd.Series:
        """
        砖型图反包力度因子（纯反包力度版）
        
        只计算反包力度，不包含其他复合逻辑
        反包力度 = sqrt(今日红柱 / 昨日绿柱)
        """
        df = df.copy()

        # 计算砖型图（使用正确的SMA）
        df['HHV4'] = df['high'].rolling(window=4, min_periods=1).max()
        df['LLV4'] = df['low'].rolling(window=4, min_periods=1).min()
        
        df['VAR1A'] = (df['HHV4'] - df['close']) / (df['HHV4'] - df['LLV4']).replace(0, np.nan) * 100 - 90
        df['VAR1A'] = df['VAR1A'].replace([np.inf, -np.inf], 0).fillna(0)
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
        df['昨日绿柱'] = df['砖型图_前2'] - df['砖型图_前1']
        df['昨日绿柱'] = df['昨日绿柱'].where(df['砖型图_前2'] > df['砖型图_前1'], 0)
        df['今日红柱'] = df['砖型图'] - df['砖型图_前1']
        df['今日红柱'] = df['今日红柱'].where(df['砖型图'] > df['砖型图_前1'], 0)

        # 只计算反包力度
        df['反包比例'] = df['今日红柱'] / df['昨日绿柱'].replace(0, np.nan)
        df['反包比例'] = df['反包比例'].replace([np.inf, -np.inf], 0).fillna(0)
        df['反包力度'] = np.sqrt(df['反包比例'].clip(lower=0))

        return df['反包力度']