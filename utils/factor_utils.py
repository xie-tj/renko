import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FactorUtils:
    """因子预处理工具"""
    
    @staticmethod
    def winsorize(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """
        去极值处理（缩尾）
        
        Args:
            df: DataFrame
            col: 需要处理的列名
            lower: 下分位数
            upper: 上分位数
        """
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        return df[col].clip(lower_bound, upper_bound)
    
    @staticmethod
    def standardize(df: pd.DataFrame, col: str) -> pd.Series:
        """标准化（z-score）"""
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            return pd.Series(0, index=df.index)
        return (df[col] - mean) / std
    
    @staticmethod
    def neutralize(df: pd.DataFrame, factor_col: str, group_col: str = 'industry') -> pd.Series:
        """
        行业中性化
        对每个行业内的因子进行标准化，消除行业差异
        """
        result = pd.Series(index=df.index, dtype=float)
        
        for group in df[group_col].unique():
            mask = df[group_col] == group
            group_data = df.loc[mask, factor_col]
            if len(group_data) > 1:
                result.loc[mask] = (group_data - group_data.mean()) / group_data.std()
            else:
                result.loc[mask] = 0
        
        return result
    
    @staticmethod
    def preprocess_factor(
        df: pd.DataFrame, 
        col: str,
        winsorize: bool = True,
        standardize: bool = True,
        neutralize: bool = False,
        group_col: str = 'industry'
    ) -> pd.Series:
        """
        因子预处理流水线
        
        处理顺序: 去极值 → 中性化 → 标准化
        """
        result = df[col].copy()
        
        if winsorize:
            lower = result.quantile(0.01)
            upper = result.quantile(0.99)
            result = result.clip(lower, upper)
        
        if neutralize and group_col in df.columns:
            # 行业中性化
            neutralized = pd.Series(index=df.index, dtype=float)
            for group in df[group_col].unique():
                mask = df[group_col] == group
                group_data = result[mask]
                if len(group_data) > 1 and group_data.std() > 0:
                    neutralized.loc[mask] = (group_data - group_data.mean()) / group_data.std()
                else:
                    neutralized.loc[mask] = 0
            result = neutralized
        
        if standardize:
            mean = result.mean()
            std = result.std()
            if std > 0:
                result = (result - mean) / std
            else:
                result = pd.Series(0, index=df.index)
        
        return result
