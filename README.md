# Renko Short Strategy | 砖型图短线量化选股系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Strategy-Renko-orange.svg" alt="Strategy">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Tushare-Pro-brightgreen.svg" alt="Tushare">
</p>

<p align="center">
  <b>基于通达信砖型图指标的短线选股策略，捕捉2-3日大涨机会</b>
</p>

<p align="center">
  <a href="#-快速开始">快速开始</a> •
  <a href="#-策略原理">策略原理</a> •
  <a href="#-使用指南">使用指南</a> •
  <a href="#-回测表现">回测表现</a>
</p>

---

## 📖 项目简介

本项目是一个完整的 **A股短线量化选股系统**，核心基于通达信砖型图技术指标，结合 **VCP（Volatility Contraction Pattern）形态因子** 和 **K-Means聚类算法**，实现动态权重训练和智能选股。

### ✨ 核心特性

| 特性 | 说明 |
|:---|:---|
| 🎯 **砖型图信号** | 基于通达信砖型图反转指标，捕捉短线爆发点 |
| 🤖 **智能聚类** | K-Means自动分类市场状态，不同场景使用不同权重 |
| 📊 **全量训练** | 使用270万+历史信号训练，不采样、不丢失信息 |
| ⚡ **一键运行** | 单条命令完成数据更新、信号构建、模型训练、选股全流程 |
| 🔄 **自动更新** | 智能判断数据更新时机，避免重复下载 |
| 🛡️ **风险过滤** | 自动过滤ST、退市、数据异常等风险股票 |

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Tushare Pro Token（[免费注册](https://tushare.pro/register)）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/xie-tj/renko.git
cd renko

# 2. 创建虚拟环境
conda create -n renko python=3.11 -y
conda activate renko

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 Tushare Token
## Windows PowerShell（推荐）
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "your_token_here", "User")
## 设置后重启终端使环境变量生效

# 5. 首次全量抓取数据（只需执行一次）
python main.py init

# 6. 运行完整流程
python main.py run
```

### 快速测试

```bash
# 快速测试（只处理前100只股票）
python main.py update --max-stocks 100
python main.py run
```

---

## 📁 项目结构

```
renko-short-strategy/
├── 📄 main.py                      # 主程序入口（完整量化流程）
├── 📁 scripts/
│   └── train.py                    # 独立训练脚本
├── 📁 quant/                       # 量化核心模块
│   ├── calc_factors.py             # 因子计算引擎
│   └── factor_library/             # 因子库
│       ├── renko_factors.py        # 砖型图专用因子
│       └── vcp_factors.py          # VCP形态因子
├── 📁 strategy/
│   └── renko_strategy.py           # 砖型图策略逻辑
├── 📁 utils/
│   ├── akshare_fetcher.py          # Tushare数据获取
│   ├── csv_manager.py              # CSV数据管理
│   └── factor_utils.py             # 因子工具函数
├── 📁 data/                        # 数据目录（.gitignore）
│   ├── stocks/                     # 股票历史数据
│   └── processed/                  # 处理后数据
│       ├── signal_pool.csv         # 信号池
│       └── factor_weights.json     # 因子权重
├── 📄 requirements.txt             # Python依赖
├── 📄 .env.example                 # 环境变量示例
└── 📄 README.md                    # 本文件
```

---

## 🧠 策略原理

### 1. 砖型图信号条件

策略基于通达信砖型图公式，买入信号需同时满足5个条件：

```
CC条件      : 昨日绿柱，今日红砖（砖型图反转）
力度达标    : 今日红柱 >= 昨日绿柱 × 2/3
趋势确认    : 知行短期趋势线 > 知行多空线
价格确认    : 收盘价 > 知行多空线
红砖过滤    : 红砖大小 >= 5
```

### 2. K-Means聚类

使用K-Means将信号分为3个聚类，每个聚类独立计算权重：

| 聚类 | 特征 | 正样本率 | 策略特点 |
|:---:|:---|:---:|:---|
| **聚类0** | 高波动型 | ~57% | 适合追涨，容忍高波动 |
| **聚类1** | 弱势反弹型 | ~53% | 适合抄底，关注反转 |
| **聚类2** | 强势突破型 | ~46% | 适合突破，要求确认 |

### 3. 因子体系（7个因子）

| 因子 | 说明 | 方向 | 逻辑 |
|-----|------|:---:|:---|
| `renko_strength_entity` | 砖型图反包力度 | - | 反包力度越大越好 |
| `vcp_ratio` | 波动率收敛比 | + | 收敛越好越可能突破 |
| `pivot_proximity` | 枢轴邻近度 | + | 靠近枢轴支撑 |
| `ma_alignment` | 均线多头排列 | + | MA20>MA28>MA57 |
| `closing_range` | 收盘强度 | + | 收盘接近最高价 |
| `upper_shadow_penalty` | 上影线惩罚 | + | 无上影线（光头）最佳 |
| `volatility_5day` | 前5日波动率 | + | 前期波动小有利于突破 |

---

## 📊 回测表现

### 训练效果

- **AUC**: ~0.60（有效区分正负样本）
- **区分度**: ~18（正样本得分明显高于负样本）
- **训练样本**: 270万+ 条历史信号（2020年至今）
- **正样本定义**: 未来3日收益 ≥ 3%

### 聚类效果

```
聚类分布:
  聚类 0: 892,341条 (正508,643/负383,698) - 正样本率 57.0%
  聚类 1: 1,023,456条 (正542,432/负481,024) - 正样本率 53.0%
  聚类 2: 784,203条 (正360,733/负423,470) - 正样本率 46.0%
```

---

## 🛠️ 使用指南

### 完整流程

```bash
# 一键完整流程：更新 → 构建 → 训练 → 选股
python main.py run

# 只选Top 10
python main.py run --top 10

# 跳过数据更新（已有数据时）
python main.py run --skip-update

# 跳过训练（已有权重时）
python main.py run --skip-train
```

### 分步执行

```bash
# 1. 更新数据（每日必做）
python main.py update

# 2. 构建信号池
python main.py build

# 3. 训练因子权重
python main.py train

# 4. 执行选股
python main.py select --top 20
```

### 命令速查

| 命令 | 功能 | 常用参数 |
|:-----|:-----|:---------|
| `init` | 首次全量抓取 | `--max-stocks 500`（首次建议抓取500只以上） |
| `update` | 每日增量更新 | `--max-stocks 100`（测试用） |
| `build` | 构建信号池 | - |
| `train` | 训练因子权重 | - |
| `select` | 执行选股 | `--top 10`（选前N只） |
| `run` | 一键完整流程 | `--skip-update/--skip-build/--skip-train` |

### 输出结果

选股结果保存为 CSV 格式：

```csv
rank,code,name,score,cluster,close,future_3d_return
1,000001,平安银行,85.32,0,11.12,0.0523
2,600000,浦发银行,82.15,1,9.85,0.0312
```

---

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 获取方式 |
|:-------|:-----|:---------|
| `TUSHARE_TOKEN` | Tushare Pro API Token | [Tushare官网](https://tushare.pro) 注册获取 |

### 配置方法

**Windows (PowerShell) - 推荐:**
```powershell
[Environment]::SetEnvironmentVariable("TUSHARE_TOKEN", "your_token", "User")
# 设置后重启终端
```

**macOS/Linux:**
```bash
export TUSHARE_TOKEN="your_token"
```

**或使用 .env 文件:**
```bash
cp .env.example .env
# 编辑 .env 填入你的 Token
```

---

## 🔧 进阶使用

### 自定义因子权重

编辑 `data/processed/factor_weights.json` 可手动调整权重：

```json
{
  "0": {
    "renko_strength_entity": {
      "weight": 0.20,    // 提高权重
      "direction": -1    // -1表示反向（值越小越好）
    }
  }
}
```

### 修改聚类数量

在 `main.py` 中修改 `n_clusters` 参数：

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
```

### 调整正样本阈值

默认未来3日收益 ≥ 3% 为正样本，可在 `main.py` 中修改：

```python
df['is_positive'] = df['future_3d_return'] >= 0.03  # 3%阈值
```

---

## 📈 数据说明

### 信号池格式

```csv
date,code,close,renko_strength_entity,vcp_ratio,pivot_proximity,...
2026-04-03,000001,11.12,0.85,-0.32,0.15,...
```

### 因子权重格式

```json
{
  "0": {
    "renko_strength_entity": {
      "weight": 0.148,
      "ic": -0.047,
      "direction": -1,
      "optimal_range": [-0.42, 1.76],
      "optimal_mean": 0.67
    },
    ...
  },
  "1": {...},
  "2": {...}
}
```

---

## ❓ 常见问题

### Q1: 选股结果中股票名称显示"未知"？

**原因**: 首次抓取数据时未保存股票名称映射。

**解决**: 删除旧数据重新抓取，会自动保存名称：
```bash
# 删除旧数据
rm data/stocks/*.csv
rm data/stock_names.json

# 重新抓取
python main.py update --max-stocks 500
```

### Q2: 运行时报错 `n_samples < n_clusters`？

**原因**: 信号数量少于 K-Means 聚类数（默认3个）。

**解决**: 抓取更多股票数据，或程序会自动调整为 `min(3, n_samples)` 个聚类。

### Q3: Windows 控制台显示乱码？

**原因**: Windows 默认使用 GBK 编码，不支持部分 Unicode 字符。

**解决**: 设置 UTF-8 编码（可选，不影响程序运行）：
```powershell
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding
```

### Q4: 如何调整选股数量？

```bash
# 选股 Top 10
python main.py select --top 10

# 或完整流程时指定
python main.py run --top 10
```

### Q5: 如何只更新部分股票进行测试？

```bash
# 只更新前100只股票（快速测试）
python main.py update --max-stocks 100
```

---

## ⚠️ 风险提示

1. **本策略仅供学习研究，不构成投资建议**
2. **量化交易有风险，历史回测不代表未来收益**
3. **短线交易波动大，请根据自身风险承受能力谨慎使用**
4. **建议先进行模拟盘验证，再考虑实盘交易**

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交规范

- 使用清晰的 commit message
- 新增功能请补充文档
- 修复 bug 请说明问题现象和解决方案

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢

- [Tushare](https://tushare.pro) - 提供高质量A股数据
- [scikit-learn](https://scikit-learn.org) - 提供K-Means等机器学习算法
- [pandas](https://pandas.pydata.org) - 提供高效数据处理工具

---

<p align="center">
  <b>⭐ Star 本项目 if you find it helpful!</b>
</p>