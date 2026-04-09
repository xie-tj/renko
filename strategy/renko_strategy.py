#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
砖型图选股策略 - 通达信公式实现

信号条件：
1. 昨天绿柱，今天红柱（砖型图反转）
2. 今日红柱高度 >= 昨日绿柱高度的 2/3（力度达标）
3. 知行短期趋势线 > 知行多空线（趋势向上）
4. 收盘价 > 知行多空线（价格确认）
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.strategy_registry import BaseStrategy
from config.logger_config import get_logger

logger = get_logger(__name__)


class RenkoStrategy(BaseStrategy):
    """
    砖型图超短策略
    
    基于通达信砖型图公式，捕捉2日大涨机会
    """
    
    def __init__(self):
        # 默认参数
        params = {
            'renko_strength_ratio': 0.67,  # 反包力度要求 (2/3)
            'min_data_days': 120,          # 最小数据天数（计算MA114）
            'prediction_days': 2,          # 预测持有天数
            'min_red_brick': 5,            # 最小红砖大小阈值
        }
        
        super().__init__(name="砖型图反包", params=params)
        
        self.description = "砖型图红柱反包策略，捕捉趋势转折点的2日大涨机会"
    
    @staticmethod
    def _sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
        """
        通达信SMA平滑移动平均
        SMA = (M * X + (N-M) * 昨日SMA) / N
        """
        result = pd.Series(index=series.index, dtype=float)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            if pd.isna(series.iloc[i]):
                result.iloc[i] = result.iloc[i-1]
            else:
                result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result
    
    def analyze_stock(self, code: str, name: str, df: pd.DataFrame) -> dict:
        """
        分析单只股票
        
        Returns:
            如果有信号返回详情，否则返回None
        """
        # ========== 股票过滤条件 ==========
        # 1. 过滤 ST/*ST
        if 'ST' in name or '*ST' in name:
            return None
        
        # 2. 过滤退市/即将退市
        if '退' in name:
            return None
        
        # 数据验证
        if not self.validate_data(df, self.params['min_data_days']):
            return None
        
        # 按日期升序排列（ oldest first ）
        df = df.sort_values('date', ascending=True).copy()
        
        # 3. 过滤停牌 (成交量为0或价格无变化)
        latest_raw = df.iloc[-1]
        if latest_raw.get('volume', 0) == 0:
            return None
        
        # 计算指标
        df = self._calc_indicators(df)
        
        # 检查最新数据是否有信号
        latest = df.iloc[-1]
        
        if not latest['XG']:
            return None
        
        # 新增：过滤红砖小于阈值的信号
        if latest['今日红柱'] < self.params['min_red_brick']:
            return None
        
        # 构建信号详情
        signal = {
            'code': code,
            'name': name,
            'date': latest['date'].strftime('%Y-%m-%d') if isinstance(latest['date'], pd.Timestamp) else str(latest['date']),
            'close': round(latest['close'], 2),
            'signals': self._get_signal_tags(latest),
            'score': self._calc_score(latest),
            'indicators': {
                '砖型图值': round(latest['砖型图'], 2),
                '今日红柱': round(latest['今日红柱'], 4),  # 新增
                '知行短期趋势线': round(latest['知行短期趋势线'], 2),
                '知行多空线': round(latest['知行多空线'], 2),
                '信号强度': round(latest['信号强度'], 3) if not pd.isna(latest['信号强度']) else None,
                '趋势偏离': round(latest['趋势偏离'] * 100, 2)  # 百分比
            },
            'recommendation': '买入' if latest['XG'] else '观望'
        }
        
        return signal
    
    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()
        
        # ============== 1. 知行趋势线计算 ==============
        # 知行短期趋势线: EMA(EMA(C,10),10)
        df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['知行短期趋势线'] = df['EMA10'].ewm(span=10, adjust=False).mean()
        
        # 知行多空线: (MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4
        df['MA14'] = df['close'].rolling(window=14, min_periods=1).mean()
        df['MA28'] = df['close'].rolling(window=28, min_periods=1).mean()
        df['MA57'] = df['close'].rolling(window=57, min_periods=1).mean()
        df['MA114'] = df['close'].rolling(window=114, min_periods=1).mean()
        df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
        
        # ============== 2. 砖型图核心计算 ==============
        # VAR1A:=(HHV(HIGH,4)-CLOSE)/(HHV(HIGH,4)-LLV(LOW,4))*100-90
        df['HHV4'] = df['high'].rolling(window=4, min_periods=1).max()
        df['LLV4'] = df['low'].rolling(window=4, min_periods=1).min()
        
        df['VAR1A'] = (df['HHV4'] - df['close']) / (df['HHV4'] - df['LLV4']).replace(0, np.nan) * 100 - 90
        df['VAR1A'] = df['VAR1A'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # VAR2A:=SMA(VAR1A,4,1)+100
        df['VAR2A'] = self._sma(df['VAR1A'], 4, 1) + 100
        
        # VAR3A:=(CLOSE-LLV(LOW,4))/(HHV(HIGH,4)-LLV(LOW,4))*100
        df['VAR3A'] = (df['close'] - df['LLV4']) / (df['HHV4'] - df['LLV4']).replace(0, np.nan) * 100
        df['VAR3A'] = df['VAR3A'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # VAR4A:=SMA(VAR3A,6,1)
        df['VAR4A'] = self._sma(df['VAR3A'], 6, 1)
        
        # VAR5A:=SMA(VAR4A,6,1)+100
        df['VAR5A'] = self._sma(df['VAR4A'], 6, 1) + 100
        
        # VAR6A:=VAR5A-VAR2A
        df['VAR6A'] = df['VAR5A'] - df['VAR2A']
        
        # 砖型图:=IF(VAR6A>4,VAR6A-4,0)
        df['砖型图'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)
        
        # ============== 3. 红柱绿柱判断 ==============
        df['砖型图_前1'] = df['砖型图'].shift(1)
        df['砖型图_前2'] = df['砖型图'].shift(2)
        
        # AA:=REF(砖型图,1)<砖型图  {今天涨，出红柱}
        df['AA'] = df['砖型图_前1'] < df['砖型图']
        
        # BB:=REF(砖型图,1)>砖型图  {今天跌，出绿柱}
        df['BB'] = df['砖型图_前1'] > df['砖型图']
        
        # ============== 4. 反包力度计算 ==============
        # 昨日绿柱:=REF(砖型图,2)-REF(砖型图,1)
        df['昨日绿柱'] = df['砖型图_前2'] - df['砖型图_前1']
        
        # 今日红柱:=砖型图-REF(砖型图,1)
        df['今日红柱'] = df['砖型图'] - df['砖型图_前1']
        
        # 力度达标:=今日红柱 >= (昨日绿柱 * 2 / 3)
        strength_ratio = self.params['renko_strength_ratio']
        df['力度达标'] = df['今日红柱'] >= (df['昨日绿柱'] * strength_ratio)
        
        # ============== 5. 信号核心逻辑 ==============
        # CC:=REF(BB,1) AND AA  {昨天是绿柱，今天是红柱}
        df['BB_前1'] = df['BB'].shift(1)
        df['CC'] = df['BB_前1'] & df['AA']
        
        # ============== 6. 新增筛选条件 ==============
        # 条件1:=知行短期趋势线 > 知行多空线
        df['条件1'] = df['知行短期趋势线'] > df['知行多空线']
        
        # 条件2:=CLOSE > 知行多空线
        df['条件2'] = df['close'] > df['知行多空线']
        
        # ============== 7. 最终选股信号 ==============
        # XG: CC AND 力度达标 AND 条件1 AND 条件2
        df['XG'] = df['CC'] & df['力度达标'] & df['条件1'] & df['条件2']
        
        # ============== 8. 辅助指标 ==============
        df['信号强度'] = df['今日红柱'] / df['昨日绿柱'].replace(0, np.nan)
        df['趋势偏离'] = (df['close'] - df['知行多空线']) / df['知行多空线']
        
        return df
    
    def _get_signal_tags(self, latest: pd.Series) -> list:
        """获取信号标签列表"""
        tags = []
        
        if latest['XG']:
            tags.append('砖型图反包')
        
        if latest['条件1']:
            tags.append('趋势向上')
        
        if latest['条件2']:
            tags.append('价格确认')
        
        if latest['信号强度'] > 1:
            tags.append('强势反包')
        elif latest['信号强度'] > 0.8:
            tags.append('力度良好')
        
        return tags
    
    def _calc_score(self, latest: pd.Series) -> float:
        """计算信号得分 (0-1)"""
        score = 0.0
        
        # 基础得分（有信号）
        if latest['XG']:
            score += 0.4
        
        # 力度得分
        strength = latest.get('信号强度', 0)
        if not pd.isna(strength):
            score += min(strength * 0.2, 0.2)  # 最高0.2
        
        # 趋势偏离得分
        deviation = latest.get('趋势偏离', 0)
        if not pd.isna(deviation):
            score += min(deviation * 0.5, 0.2)  # 最高0.2
        
        # 趋势确认
        if latest['条件1']:
            score += 0.1
        if latest['条件2']:
            score += 0.1
        
        return round(min(score, 1.0), 3)


# 策略实例注释掉，在 main.py 中手动创建\n# strategy_instance = RenkoStrategy()
