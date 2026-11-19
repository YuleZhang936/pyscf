# DCC-TZ（H3C）密度计算与小分子筛选

方法已并入 **`pyscf/dcc_tz.py`**，可用 `from pyscf import dcc_tz` 导入。数值实现不依赖 `work/`。
工作区只保留两个业务代码文件：`dataset.py` 负责数据读取、筛选和密度存储；`run.py` 负责运行与提交。

```python
from pyscf import dcc_tz
case = dict(name='water', atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587')
# 在计算节点执行：
# records = dcc_tz.calculate(case, memory=6000)
```

## 当前状态

2026-11-19 已按要求取消全部 QM9 计时任务：1055663_1、1055707、1055708、1055709；延后清理任务 1055746 也已取消。
已完成的 P95 样本 1055663_0 耗时 42分53秒；其余未完成计时按取消处理，不用来估计完整计算时间。

临时旧源码、旧 Slurm 脚本、运行目录和预备 QM9 配置已移除。计时日志、状态、已完成密度和旧代码已逐文件校验后收进
`data/validation.zip` 的 `stopped-qm9-timing-20261119/`。新的数值函数与移动前 AST 逐项一致；验证包内还保存之前 10 项通过的重构检查。
没有提交新的密度计算。IQC 的运行与训练库均未修改。

只保留三个数据归档：

- `data/qm9.zip`：已有 QM9 输入，留作备用。
- `data/validation.zip`：历史计算、验证及已停止计时的归档。
- `data/ixc2-small.zip`：本次筛选报告、输入构型、来源校验值及筛选代码快照；没有新计算的密度。

## 从当前 IXC2 训练库筛选

已由活动协调器 978120 定位到 `iqc/work/ixc2/stage1/runs/relative-df-20261114-quota-v686/run.json`，
核验其指定的 `iqc/work/ixc2/data/stage1.sqlite`，SHA256 为
`a0a9c59bbb09b2aa13ef0feee1a3c69439b7d53e48ef69682f9767376c32542c`。
当前配置的训练标签为 TAE25 65,594 条、BSE49 3,909 条、D442×10 2,870 条。

筛选以 **train 能量观测的完整组成体系** 为单位：母分子、碎片或二聚体和单体均须满足大小限制；
D442 曲线保留全部 10 个距离点。以下按现有 state ID 去重统计分子构型/电子态，排除另计的 18 个原子态。
不将能量标签、优化器重复访问或三个 H3C 分量当成独立密度样本。
重原子定义为 H 以外的原子，二聚体计算两侧总数。

| 上限 | 闭壳层构型 | 开壳层构型 | 完整 D442 曲线 |
| --- | ---: | ---: | ---: |
| 2 个重原子 | 405 | 71 | 11 |
| 3 个重原子 | 3,759 | 296 | 21 |
| 4 个重原子 | 56,737 | 781 | 40 |
| 3 个重原子且 aTZ ≤150 AO | 420 | 22 | 12 |
| **3 个重原子且 aTZ ≤200 AO** | **1,829** | **87** | **17** |
| 3 个重原子且 aTZ ≤250 AO | 3,231 | 176 | 19 |

粗体方案为严格小分子基础池。1,829 个闭壳层构型由 1,499 个 TAE25 分子、115 个 BSE49 母分子、170 个 D442 二聚体距离点及其 45 个单体态组成。
实际最大 aTZ 为 196 AO，aDZ 为 99 AO。这是控制计算规模的筛选标准，不是未经实测的耗时或精度保证。
这些构型合计覆盖 H–Ar 元素，但不保证各元素/相互作用均匀分布。

**更建议第一批用 2,239 个闭壳层构型的替代清单**：TAE25/BSE49 保持上述限制，只将弱相互作用复合物改为整体 aTZ ≤300 AO，不再限制其总重原子数。
二聚体包含两套单体基组，单独给它较宽的预算可以扩充相互作用覆盖，同时避免引入数万条更大的 TAE25 分子。
这会保留 41 条完整 D442 曲线（410 个二聚体距离点），连同 215 个非原子单体态和前述 1,614 个 TAE25/BSE49 构型，合计 2,239 个。
其中 15 条曲线不涉及稀有气体，新增了甲胺–氮气、二氧化碳–二硫化碳等小复合物；仍不能替代专门的氢键集。

87 个开壳层分子态中有 27 个也被原库的 validation/test 观测引用，已单列；可独立用于新增密度训练的开壳层候选为 60 个。
1,829 个闭壳层候选没有这种 state ID 重叠。去重沿用数据库的几何、电荷、自旋身份，未宣称做了旋转/原子置换下的完全几何去重。

`ixc2-small.zip` 内：

- `report.json`：全部阈值统计、基组 AO 数和 17 条曲线的体系名称。
- `cases/closed_shell.json`：1,829 个闭壳层候选。
- `cases/closed_shell_dispersion300.json`：推荐的 2,239 个闭壳层候选，包含前述基础池，不是另加 2,239 个。
- `cases/open_shell.json`：60 个不与原 held-out 观测共享的开壳层候选。
- `cases/shared_with_heldout.json`：27 个共享分子态。
- `cases/atomic_references.json`：18 个原子参考态，另计。
- `selected-energy-observations.jsonl`：1,787 条原能量观测及组成关系，可追溯每个构型。
- `expanded-dispersion-energy-observations.jsonl`：扩充弱相互作用方案对应的 2,027 条原能量观测。

当前候选仍有覆盖缺口：BSE49 提供平衡母分子与自由基端点，并不提供连续共价断键路径
（[原论文](https://www.nature.com/articles/s41597-021-01088-2)）；D442×10 侧重 London 色散
（[原论文](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp01602h)）。
在本次很小的 17 条曲线中，15 条涉及稀有气体，另两条为 H2–H2 和 H2–N2。
要覆盖常规相互作用，后续仍应补充小型氢键、偶极和有机分子色散复合物；不能把当前筛选宣称为全面覆盖。

## 使用

在仓库根目录：

```bash
source .local/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python -m work.run inspect-ixc2 --training-run /home/xiaoyuzhang/workspace/iqc/work/ixc2/stage1/runs/relative-df-20261114-quota-v686
```

这只读源数据库、统计并导出 ZIP，不提交量子计算。可用 `--max-heavy` 和 `--max-nao` 改变候选范围。
直接读取候选无需解压：

```python
from work.dataset import read_cases
cases = read_cases('work/data/ixc2-small.zip::cases/closed_shell_dispersion300.json')
```

单分子运行命令（仅在计算节点执行；本次未运行）：

```bash
python -m work.run build --manifest work/data/ixc2-small.zip::cases/closed_shell_dispersion300.json --index 0 --output work/runs/small/example.npz --threads 4
```

QM9 的 `prepare / submit / report` 批处理功能仍保留；`submit` 默认只预览，需显式 `--execute` 才提交。
小分子候选目前是筛选清单，未提交批量计算。

## 固定方法

```text
DCC-TZ = AE RI-MP2/aug-cc-pVTZ
       + FC CCSD/aug-cc-pVDZ
       - FC RI-MP2/aug-cc-pVDZ
```

三个分量均为轨道响应后的密度；没有拟合参数。冻结核数：H–He 0、Li–Ne 1、Na–Ar 5。
采用已验证的 PySCF 默认收敛设置，HF 稳定性分析已移除，仍检查求解收敛、矩阵电子数/对称性及网格积分电子数。
方法保留 R/U 路径；批量 `packed64` 压缩目前只支持闭壳层，开壳层可用保留自旋矩阵的原始分量格式。

闭壳层最终存 aTZ 的 MP2 密度和 aDZ 的 CCSD−MP2 修正两个 AO 矩阵，float64 下三角无损压缩。
不做基组间投影、占据数截断或电荷重归一化；实空间密度按需重建。
`build` 输出便于审计的三个原始分量；`dataset.write_compact` 转成上述两矩阵格式。
数值检查不保证任意分子的 eta < 1；历史独立验证的 C3 失败保存在验证归档中。

本地 `pyscf/mp/dfmp2_native.py` 的已验证维数修复仍保留：MO 数取系数矩阵列数，以支持线性相关轨道被去除后 AO 与 MO 数不等的情况。
