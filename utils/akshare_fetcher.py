#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare数据获取器 - 使用Tushare Pro接口获取股票数据

主要数据源：
1. Tushare Pro接口 - 包含换手率、市值等完整字段
"""
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime, timedelta
import time
import os
import sys
import json
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 禁用代理（避免连接问题）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# Tushare Token - 从环境变量读取（必须设置）
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    raise ValueError(
        "请设置 TUSHARE_TOKEN 环境变量\n"
        "PowerShell 设置方法: [Environment]::SetEnvironmentVariable('TUSHARE_TOKEN', '你的Token', 'User')\n"
        "设置后请重启终端使环境变量生效"
    )


class TushareFetcher:
    """Tushare数据获取器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.stocks_dir = self.data_dir / "stocks"
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Tushare Pro
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        logger.info("Tushare Pro API 初始化成功")
        
        # 更新缓存文件
        self.update_cache_file = self.data_dir / ".update_cache.json"
    
    def get_all_stock_codes(self) -> dict:
        """获取所有A股股票代码"""
        try:
            df = self.pro.stock_basic(exchange='', list_status='L', 
                                      fields='ts_code,symbol,name,area,industry,list_date')
            if df is not None and not df.empty:
                # 转换为 {code: name} 格式
                stock_dict = dict(zip(df['symbol'], df['name']))
                logger.info(f"获取到 {len(stock_dict)} 只股票")
                return stock_dict
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
        
        return {}
    
    def fetch_stock_history(self, code: str, years: int = 6, adj: str = 'qfq') -> pd.DataFrame:
        """
        获取单只股票历史数据
        
        Args:
            code: 股票代码（如 '000001'）
            years: 获取年数
            adj: 复权方式，'qfq'=前复权, 'hfq'=后复权
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount, turnover, market_cap
        """
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y%m%d')
            
            # Tushare Pro接口需要 ts_code 格式（如 '000001.SZ'）
            ts_code = self._to_ts_code(code)
            
            # 获取前复权数据（Tushare daily接口默认返回前复权）
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 如果需要后复权，获取复权因子并计算
            if adj == 'hfq':
                try:
                    adj_df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if adj_df is not None and not adj_df.empty:
                        df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
                        # 计算后复权价格
                        for col in ['open', 'high', 'low', 'close']:
                            if col in df.columns:
                                df[col] = df[col] * df['adj_factor']
                except Exception as e:
                    logger.warning(f"获取 {code} 后复权因子失败: {e}")
                    return pd.DataFrame()
            
            # 获取额外字段（换手率、市值等）
            try:
                df_daily = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df_daily is not None and not df_daily.empty:
                    df = df.merge(df_daily[['trade_date', 'turnover_rate', 'total_mv']], 
                                  on='trade_date', how='left')
            except:
                pass
            
            # 重命名列
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
                'turnover_rate': 'turnover',
                'total_mv': 'market_cap'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 按日期倒序排列
            df = df.sort_values('date', ascending=False)
            
            # 选择需要的列
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap']
            df = df[[c for c in columns if c in df.columns]]
            
            return df
            
        except Exception as e:
            logger.warning(f"获取 {code} 数据失败: {e}")
            return pd.DataFrame()
    
    def _to_ts_code(self, code: str) -> str:
        """转换为Tushare格式的ts_code"""
        code = str(code).strip()
        if code.startswith('6'):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"
    
    def init_full_data(self, max_stocks: int = None, delay: float = 0.5):
        """首次全量抓取"""
        print("=" * 60)
        print(" 开始全量抓取")
        print("=" * 60)
        
        stock_dict = self.get_all_stock_codes()
        if not stock_dict:
            print("无法获取股票列表")
            return
        
        stock_codes = list(stock_dict.keys())
        if max_stocks:
            stock_codes = stock_codes[:max_stocks]
        
        total = len(stock_codes)
        success = 0
        failed = 0
        
        # 导入 CSVManager 用于保存股票名称
        from utils.csv_manager import CSVManager
        csv_manager = CSVManager(str(self.data_dir))
        
        for i, code in enumerate(stock_codes, 1):
            name = stock_dict.get(code, '')
            
            # 显示进度条（使用ASCII字符避免编码问题）
            progress = int((i / total) * 40)
            bar = '#' * progress + '-' * (40 - progress)
            print(f"\r[{bar}] {i}/{total} {code} {name[:8]:<8s}", end='', flush=True)
            
            try:
                df = self.fetch_stock_history(code, years=6)
                if df is not None and not df.empty and len(df) >= 10:
                    file_path = self.stocks_dir / f"{code}.csv"
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    # 保存股票名称
                    if name:
                        csv_manager.set_stock_name(code, name)
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
            
            if i % 10 == 0:
                time.sleep(delay)
        
        print()  # 换行
        print("=" * 60)
        print(f" 完成! 成功: {success}, 失败: {failed}")
        print("=" * 60)
    
    def daily_update(self, max_stocks: int = None):
        """每日增量更新"""
        print("=" * 60)
        print(" 每日增量更新")
        print("=" * 60)
        
        from utils.csv_manager import get_csv_manager
        csv_manager = get_csv_manager(str(self.data_dir))
        
        existing = csv_manager.list_all_stocks()
        if not existing:
            print("没有现有数据，请先执行 init")
            return
        
        if max_stocks:
            existing = existing[:max_stocks]
        
        today = datetime.now().date()
        current_time = datetime.now().time()
        market_close = datetime.strptime("15:00", "%H:%M").time()
        
        # 判断更新目标日期
        if current_time < market_close:
            target_date = today - timedelta(days=1)
            print(f" 当前时间 {current_time.strftime('%H:%M')}，尚未收盘，更新至 {target_date}")
        else:
            target_date = today
            print(f" 当前时间 {current_time.strftime('%H:%M')}，已收盘，更新至今天")
        
        # 检查缓存
        update_cache = {}
        if self.update_cache_file.exists():
            try:
                with open(self.update_cache_file, 'r') as f:
                    update_cache = json.load(f)
            except:
                pass
        
        target_date_str = target_date.strftime('%Y-%m-%d')
        if update_cache.get('last_update_date') == target_date_str and not max_stocks:
            print(f" 数据已于 {target_date_str} 更新过")
            return
        
        total = len(existing)
        updated = 0
        failed = 0
        failed_codes = []  # 记录失败的股票代码
        
        for i, code in enumerate(existing, 1):
            # 显示进度条（使用ASCII字符避免编码问题）
            progress = int((i / total) * 40)
            bar = '#' * progress + '-' * (40 - progress)
            print(f"\r[{bar}] {i}/{total} {code} 更新中...", end='', flush=True)
            
            try:
                df_new = self.fetch_stock_history(code, years=1)
                if df_new is not None and not df_new.empty:
                    file_path = self.stocks_dir / f"{code}.csv"
                    
                    if file_path.exists():
                        df_existing = pd.read_csv(file_path)
                        df_existing['date'] = pd.to_datetime(df_existing['date'])
                        
                        df_combined = pd.concat([df_existing, df_new]).drop_duplicates('date', keep='last')
                        df_combined = df_combined.sort_values('date', ascending=False)
                    else:
                        df_combined = df_new
                    
                    df_combined.to_csv(file_path, index=False, encoding='utf-8-sig')
                    updated += 1
                else:
                    # 数据为空，可能是停牌或退市，忽略不记录为失败
                    logger.debug(f"获取 {code} 数据为空，可能停牌或退市，忽略")
            except Exception as e:
                failed += 1
                failed_codes.append(code)
                logger.error(f"获取 {code} 数据失败: {e}")
        
        print("=" * 60)
        print(f" 更新完成! 成功: {updated}, 失败: {failed}")
        
        # 重试失败的股票
        if failed_codes:
            print(f"\n 开始重试 {len(failed_codes)} 只失败的股票...")
            print(" 等待5秒让API限流重置...")
            time.sleep(5)  # 等待更长时间
            
            retry_success = 0
            retry_failed = 0
            retry_failed_codes = []  # 记录仍然失败的股票
            
            for i, code in enumerate(failed_codes, 1):
                progress = int((i / len(failed_codes)) * 30)
                bar = '#' * progress + '-' * (30 - progress)
                print(f"\r   重试 [{bar}] {i}/{len(failed_codes)} {code}", end='', flush=True)
                
                try:
                    # 每10只股票等待一下，避免触发限流
                    if i % 10 == 0:
                        time.sleep(1)
                    
                    df_new = self.fetch_stock_history(code, years=1)
                    if df_new is not None and not df_new.empty:
                        file_path = self.stocks_dir / f"{code}.csv"
                        
                        if file_path.exists():
                            df_existing = pd.read_csv(file_path)
                            df_existing['date'] = pd.to_datetime(df_existing['date'])
                            
                            df_combined = pd.concat([df_existing, df_new]).drop_duplicates('date', keep='last')
                            df_combined = df_combined.sort_values('date', ascending=False)
                        else:
                            df_combined = df_new
                        
                        df_combined.to_csv(file_path, index=False, encoding='utf-8-sig')
                        retry_success += 1
                    else:
                        # 数据为空，忽略不记录为失败
                        pass
                except Exception as e:
                    retry_failed += 1
                    retry_failed_codes.append(code)
                    logger.error(f"重试 {code} 仍然失败: {e}")
            
            print()  # 换行
            print(f" 重试结果: 成功 {retry_success}")
            
            # 显示仍然失败的股票（只有真正出错的股票）
            if retry_failed_codes:
                print(f" 仍然失败的股票: {len(retry_failed_codes)} 只")
                print(f" 示例: {', '.join(retry_failed_codes[:10])}")
            
            updated += retry_success
        
        print("=" * 60)
        print(f" 最终统计: 成功 {updated}, 失败 {failed}")
        print("=" * 60)


# 保持向后兼容的别名
AKShareFetcher = TushareFetcher
