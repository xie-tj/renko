import pandas as pd
import numpy as np
from typing import List
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.factor_library.renko_factors import RenkoFactors
from quant.factor_library.vcp_factors import VCPFactors

logger = logging.getLogger(__name__)


class FactorCalculator:
    """
    【重构版】VCP形态因子计算器
    
    仅保留renko_strength_entity，其余替换为VCP因子体系
    """
    
    FACTOR_LIST = [
        # === 保留：砖型图专用因子 ===
        ('renko_strength_entity', RenkoFactors.renko_strength_entity, {}),
        
        # === 新增：VCP形态因子 ===
        # 1. 形态总分核心因子
        ('vcp_ratio', VCPFactors.volatility_contraction, {'short_window': 5, 'long_window': 35}),
        ('pivot_proximity', VCPFactors.pivot_proximity, {'window': 35}),
        ('ma_alignment', VCPFactors.ma_alignment, {}),
        # 2. 当日情绪加分项
        ('closing_range', VCPFactors.closing_range, {}),
        # 3. P惩罚项
        ('upper_shadow_penalty', VCPFactors.upper_shadow_penalty, {}),
        # 4. 前5日波动率
        ('volatility_5day', VCPFactors.volatility_5day, {}),
    ]
    
    @classmethod
    def calc_all_factors(cls, df: pd.DataFrame, code: str = None, show_progress: bool = False) -> pd.DataFrame:
        """
        批量计算所有因子
        
        Args:
            df: 包含日线数据的DataFrame
            code: 股票代码（可选，用于groupby）
            show_progress: 是否显示进度条
        
        Returns:
            添加了因子列的DataFrame
        """
        df = df.copy()
        
        # 如果没有code列，添加一个临时列
        if 'code' not in df.columns:
            if code:
                df['code'] = code
            else:
                df['code'] = 'unknown'
        
        total_factors = len(cls.FACTOR_LIST)
        
        for i, (factor_name, factor_func, kwargs) in enumerate(cls.FACTOR_LIST, 1):
            try:
                df[factor_name] = factor_func(df, **kwargs)
                if show_progress:
                    # 打印进度条
                    progress = int((i / total_factors) * 30)  # 30个字符的进度条
                    bar = '█' * progress + '░' * (30 - progress)
                    print(f"\r  计算因子 [{bar}] {i}/{total_factors} {factor_name:<25s}", end='', flush=True)
            except Exception as e:
                logger.error(f"因子 {factor_name} 计算失败: {e}")
                df[factor_name] = np.nan
        
        if show_progress:
            print()  # 换行
        
        # 因子预处理（去极值、标准化）
        df = cls._preprocess_factors(df)
        
        return df
    
    @staticmethod
    def _preprocess_factors(df: pd.DataFrame) -> pd.DataFrame:
        """对因子进行预处理"""
        from utils.factor_utils import FactorUtils
        
        factor_cols = [f[0] for f in FactorCalculator.FACTOR_LIST]
        
        # 类别型因子不标准化（保持原始阈值含义）
        categorical_factors = ['ma_alignment', 'bullish_reversal']
        
        logger.info(f"开始预处理 {len(factor_cols)} 个因子")
        
        for col in factor_cols:
            if col not in df.columns:
                logger.warning(f"因子 {col} 不存在，跳过")
                continue
            
            # 记录预处理前的统计
            orig_mean = df[col].mean()
            orig_std = df[col].std()
            
            if col in categorical_factors:
                # 类别型因子：只去极值，不标准化
                df[col] = FactorUtils.preprocess_factor(
                    df, col, 
                    winsorize=True, 
                    standardize=False,
                    neutralize=False
                )
                logger.info(f"{col}: 类别型因子，仅去极值")
            else:
                # 连续型因子：去极值 + 标准化
                df[col] = FactorUtils.preprocess_factor(
                    df, col, 
                    winsorize=True, 
                    standardize=True,
                    neutralize=False
                )
                # 记录预处理后的统计
                new_mean = df[col].mean()
                new_std = df[col].std()
                logger.info(f"{col}: 均值 {orig_mean:.4f}→{new_mean:.4f}, 标准差 {orig_std:.4f}→{new_std:.4f}")
        
        return df
    
    @staticmethod
    def get_factor_names() -> List[str]:
        """获取所有因子名称"""
        return [f[0] for f in FactorCalculator.FACTOR_LIST]
