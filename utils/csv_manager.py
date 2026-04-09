#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据管理器 - 统一管理本地股票数据

功能：
1. 扫描和管理本地CSV文件
2. 提供统一的数据读取接口
3. 支持增量更新和缓存
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import logging

# 配置日志
logger = logging.getLogger(__name__)


class CSVManager:
    """
    【核心】CSV数据管理器
    
    数据目录结构：
    data/
    ├── stocks/              # 股票日线数据
    │   ├── 000001.csv      # 股票代码.csv
    │   ├── 000002.csv
    │   └── ...
    ├── index/               # 指数数据
    │   ├── sh000001.csv    # 上证指数
    │   └── sz399001.csv    # 深证成指
    ├── stock_names.json     # 股票代码-名称映射
    └── last_update.json     # 最后更新时间记录
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.stocks_dir = self.data_dir / "stocks"
        self.index_dir = self.data_dir / "index"
        
        # 创建目录
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 股票名称映射缓存
        self._stock_names = None
        
        logger.info(f"CSV管理器初始化: {self.data_dir}")
    
    def list_all_stocks(self) -> List[str]:
        """获取所有股票代码列表"""
        if not self.stocks_dir.exists():
            return []
        
        stocks = []
        for csv_file in self.stocks_dir.glob("*.csv"):
            code = csv_file.stem
            stocks.append(code)
        
        return sorted(stocks)
    
    def read_stock(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        读取单只股票数据
        
        Args:
            code: 股票代码 (如 '000001')
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, amount, ...]
        """
        file_path = self.stocks_dir / f"{code}.csv"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # 标准化列名
            df.columns = [c.lower().strip() for c in df.columns]
            
            # 确保date列存在并转换
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'trade_date' in df.columns:
                df['date'] = pd.to_datetime(df['trade_date'])
            
            # 按日期降序排列（最新的在前面）
            df = df.sort_values('date', ascending=False)
            
            # 日期过滤
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['date'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df['date'] <= end_dt]
            
            return df
            
        except Exception as e:
            logger.error(f"读取股票 {code} 数据失败: {e}")
            return pd.DataFrame()
    
    def save_stock(self, code: str, df: pd.DataFrame, merge: bool = True):
        """
        保存股票数据
        
        Args:
            code: 股票代码
            df: 股票数据DataFrame
            merge: 是否合并现有数据（去重）
        """
        file_path = self.stocks_dir / f"{code}.csv"
        
        try:
            # 重置索引避免重复索引问题
            df = df.reset_index(drop=True)
            
            if merge and file_path.exists():
                # 读取现有数据
                existing_df = pd.read_csv(file_path)
                existing_df.columns = [c.lower().strip() for c in existing_df.columns]
                existing_df = existing_df.reset_index(drop=True)
                
                # 合并并去重
                combined = pd.concat([existing_df, df], ignore_index=True)
                if 'date' in combined.columns:
                    combined['date'] = pd.to_datetime(combined['date'])
                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                    combined = combined.sort_values('date', ascending=False)
                    combined = combined.reset_index(drop=True)
                
                combined.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            # 静默处理错误，避免刷屏
            pass
    
    def get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        if self._stock_names is None:
            self._load_stock_names()
        
        return self._stock_names.get(code, '未知')
    
    def set_stock_name(self, code: str, name: str):
        """设置股票名称"""
        if self._stock_names is None:
            self._load_stock_names()
        
        self._stock_names[code] = name
        self._save_stock_names()
    
    def _load_stock_names(self):
        """加载股票名称映射"""
        names_file = self.data_dir / "stock_names.json"
        
        if names_file.exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                self._stock_names = json.load(f)
        else:
            self._stock_names = {}
    
    def _save_stock_names(self):
        """保存股票名称映射"""
        names_file = self.data_dir / "stock_names.json"
        
        with open(names_file, 'w', encoding='utf-8') as f:
            json.dump(self._stock_names, f, ensure_ascii=False, indent=2)
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要统计"""
        stocks = self.list_all_stocks()
        
        if not stocks:
            return {
                'total_stocks': 0,
                'date_range': None,
                'latest_date': None
            }
        
        # 采样检查日期范围
        all_dates = []
        for code in stocks[:50]:  # 采样前50只
            df = self.read_stock(code)
            if not df.empty:
                all_dates.append(df['date'].min())
                all_dates.append(df['date'].max())
        
        if all_dates:
            return {
                'total_stocks': len(stocks),
                'date_range': {
                    'start': min(all_dates).strftime('%Y-%m-%d'),
                    'end': max(all_dates).strftime('%Y-%m-%d')
                },
                'latest_date': max(all_dates).strftime('%Y-%m-%d')
            }
        
        return {
            'total_stocks': len(stocks),
            'date_range': None,
            'latest_date': None
        }
    
    def export_to_unified_format(self, output_file: str = None) -> pd.DataFrame:
        """
        导出为统一格式的合并数据
        
        Returns:
            包含所有股票的DataFrame，格式：[date, code, name, open, high, low, close, volume, amount]
        """
        stocks = self.list_all_stocks()
        all_data = []
        
        logger.info(f"开始导出 {len(stocks)} 只股票的数据...")
        
        for code in stocks:
            df = self.read_stock(code)
            if not df.empty:
                df['code'] = code
                df['name'] = self.get_stock_name(code)
                all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            # 标准化列
            cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount']
            available_cols = [c for c in cols if c in combined.columns]
            combined = combined[available_cols]
            
            # 保存
            if output_file:
                output_path = self.data_dir / output_file
                combined.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"统一格式数据已保存: {output_path}")
            
            return combined
        
        return pd.DataFrame()


# 全局实例（单例模式）
_csv_manager_instance = None


def get_csv_manager(data_dir: str = "data") -> CSVManager:
    """获取CSV管理器实例（单例）"""
    global _csv_manager_instance
    
    if _csv_manager_instance is None:
        _csv_manager_instance = CSVManager(data_dir)
    
    return _csv_manager_instance
