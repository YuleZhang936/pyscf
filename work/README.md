# DCC-TZ（H3C）与小分子密度输入

方法在 `pyscf/dcc_tz.py`，可直接 `from pyscf import dcc_tz`，不依赖 `work/`。
工作区只有两个业务代码文件：`dataset.py` 处理输入、筛选与密度存储，`run.py` 处理运行与提交。

**已准备 3,898 个闭壳层分子构型的输入；未提交量子计算。** 数据在 `data/small-density.zip`，
包含坐标、电荷、自旋、片段、来源、筛选报告、原始数据及许可证，可直接供批处理读取。
这些是待计算的输入，不是已有高精度密度标签，也不是新建的独立精度测试集。

## 数据选择

保留当前 IXC2 训练库的小分子基础池，再从 NCI Atlas 和 A24 补充相互作用。
下表“构型数”包括需要计算的非原子单体，已按精确笛卡尔坐标、电荷、自旋去重。
不同取向、不同原子排序或仅近似相同的几何不合并；不等于不同化学式的数量。

| 来源 | 主要内容 | 所选复合物几何 | 默认计算构型数 |
| --- | --- | ---: | ---: |
| TAE25 | H–Ar 共价小分子、平衡构型 | — | 1,499 |
| BSE49 | 小分子母体；自由基端点另列 | — | 115 |
| D442×10 | 色散、弱相互作用 | 41 条曲线 ×10 点 | 625 |
| HB375×10 | O/N/C–H 中性氢键及原集 noHB 对照 | 17×10 | 204 |
| IHB100×10 | 阳离子/阴离子参与的氢键 | 21×10 | 252 |
| HB300SPX×10 | 扩展到 S/P/F/Cl 的氢键 | 53×10 | 636 |
| SH250×10 | 卤键、硫族键、磷族键 | 9×10 | 108 |
| R739×5 | 短程排斥、不同元素接触 | 66×5 | 387 |
| A24 | 小分子偶极、混合相互作用、有机色散、非芳香 π 接触 | 24 个原始构型 | 70 |
| IL16 | 正负离子对：乙铵–氯离子；Cl⁻ 原子参考另列 | 1 个原始构型 | 2 |
| **合计** | **207 条完整非共价曲线 +24 个 A24 构型 +1 个离子对** | **1,765 个复合物构型** | **3,898** |

总数为 2,239 个 IXC2 基础构型、1,355 个新增复合物和 343 个新增单体记录，去除 39 个精确重复。
新增曲线的单体只取原始 scaling=1.0 几何，不优化、不加 ghost 原子；片段内部距离沿曲线的最大变化
小于 1.42×10⁻⁹ Å（源文件舍入）。不把三个密度分量计作三个样本。

筛选规则固定在 `dataset.prepare_small`：

- 元素限定 H–Ar；默认批次 spin=0。电荷分布为 0：3,666，+1：45，−1：176，−2：11。
- TAE25/BSE49：≤3 个重原子且 aTZ ≤200 AO；D442：完整复合物 aTZ ≤300 AO。
- 新 NCI 集首先保留 aTZ ≤300 AO 的完整曲线。对仍缺失的类别，最多补一个最小代表，且 ≤400 AO。
  补充为 1 条 NH–O 中性氢键、4 条阳离子氢键、7 条 σ-hole 曲线；降低阈值会丢掉这些类别。
- R739 对每个 `(原始 group, 接触元素对)` 只保留最小的一个完整五点曲线，避免同类构型过量。
- IL16 的 16 个离子对中仅乙铵–氯离子在 400 AO 以内（372 AO），纳入一个原始构型及乙铵单体。
- A24 保留全部 24 个原始几何。其分组由本项目整理，不表示相互作用具有纯粹的单一物理成分。

实际 aTZ AO 数中位数 **188**，P95 **299**，最大 **395**；129 个构型超过 300 AO，占 3.31%。
最大 aDZ 为 192 AO，最多 13 个原子。这些是规模指标，不是总耗时预测；本轮未恢复计时任务。

覆盖限于上述小分子类型：没有芳香环堆积、过渡金属、一般反应过渡态或溶剂构象系综。
SH250 的 Cl–芳香碳类别因最小体系也有 514 个 aTZ AO 而未选入。
IHB 提供带电氢键，IL16 补充一个真正的正负离子对；离子对的化学多样性仍有限。
BSE49 提供母分子/自由基端点，不是连续断键路径。

另外备好 **60 个开壳层候选**和 **80 个派生双原子键长扫描点**，均不进入默认批次。
扫描母体为 H₂、LiH、HF、HCl、N₂、CO、F₂、Cl₂，按源键长的
0.8、0.9、1、1.1、1.25、1.5、1.75、2、2.5、3 倍生成，保留源母体 ID。
这是 singlet 分支的研究挑战输入；深度拉伸可能需要多参考方法，不能直接保证 H3C 精度。

## 来源与可追溯性

还检查了 GMTKN55 的 IL16 集。其余 15 个离子对为 464–694 个 aTZ AO，超出本批预算；
报告记录全部 16 个体系的大小，包内保留用于核对的原始几何。

- [TAE25 原始数据](https://huggingface.co/datasets/microsoft/msr-acc-tae25)：CDLA-Permissive-2.0；[原论文](https://arxiv.org/abs/2506.14492)。
- [GMTKN55 作者仓库](https://github.com/grimme-lab/GMTKN55)：CC BY 4.0，固定 revision `8d485b37a1ca8837e395042671ca5ba4e0714691`；
  [GMTKN55 原论文](https://doi.org/10.1039/C7CP04913G)、[IL16 原论文](https://doi.org/10.1021/ct5010593)。
- [BSE49 原论文](https://www.nature.com/articles/s41597-021-01088-2)。
- [NCI Atlas 作者仓库](https://github.com/Honza-R/NCIAtlas)：CC BY 4.0；固定 revision `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`。
  原 README 列出每个子集的作者和论文，完整保存在输入 ZIP 中。
- [A24 原论文](https://doi.org/10.1021/ct400057w)；几何取自 [Psi4 A24 模块](https://github.com/psi4/psi4/blob/c3cd35420200a5c7122930b517c339056e763c85/psi4/share/psi4/databases/A24.py)。
  保留未修改的模块及 LGPL-3.0 许可证；仅解析字面量，不执行下载的源码。

IXC2 基础池来自活动配置 `relative-df-20261114-quota-v686` 指定的训练库，数据库 SHA256：
`a0a9c59bbb09b2aa13ef0feee1a3c69439b7d53e48ef69682f9767376c32542c`。
基础闭壳层池没有与该库 held-out 观测共享的 state ID；27 个共享的开壳层态单列在原筛选归档。
新增公开数据尚未分配 train/test，且可能与其他基准有来源或几何重叠。
将来分割训练和验证时，应连同曲线、相关单体一起分组，再检查跨来源重叠。

`small-density.zip` 内：

- `cases/production.json` 与 `dcc/inputs/*.jsonl`：同一批 3,898 个输入，后者预分成 64 个调度分片。
- `cases/open_shell.json`、`cases/bond_stretch_challenge.json`：默认队列之外的候选。
- `cases/atomic_references.json`：原子参考记录，另计，不在分子计算数中。
- `report.json`：选取数量、缺失类别、大小分布、来源校验值及规则。
- `selection/curves.json`、`selection/exact_aliases.json`：完整曲线与重复记录映射。
- `sources/`：未修改的公开源文件，以及原 `ixc2-small.zip` 的筛选、能量观测、许可证和来源快照。
- `ATTRIBUTION.txt`、`index.json`：出处说明及逐文件 SHA256。

其他两个 ZIP 为 `qm9.zip`（备用输入）和 `validation.zip`（历史验证与已停止计时），均不推送到 Git。
原独立 `ixc2-small.zip` 完整收进新输入包，避免重复保留。

## 使用

在仓库根目录激活环境。以下命令只准备配置和预览，不提交计算：

```bash
source .local/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python -m work.run prepare
python -m work.run submit --concurrency 4
```

默认输入 `work/data/small-density.zip`，默认输出 `work/runs/small-density`。
64 个分片按 AO 数交错分配，每个作业内依次计算多个分子并保存检查点；不是每个分子提交一个作业。
每个分子严格使用一个节点、一个进程。只有显式 `submit --execute` 才调用 Slurm 提交。
当前资源模板沿用每作业 52 线程、96 GiB，尚未针对新数据池调优。

离线重建输入包（使用包内已校验的原始数据，不依赖 IQC 数据库或网络）：

```bash
python -m work.run prepare-small
```

单个候选读取无需解压：

```python
from work.dataset import read_cases
cases = read_cases('work/data/small-density.zip::cases/production.json')
```

`build --manifest ZIP::cases/production.json --index N --output FILE --threads N` 用于计算节点上的单分子计算。
备用 QM9 可用 `prepare --input work/data/qm9.zip --root work/runs/qm9` 显式指定。
旧 QM9 计时任务在 2026-11-19 已取消，日志与结果归入 `validation.zip/stopped-qm9-timing-20261119/`；IQC 训练未改动。

## 固定方法与存储

```text
DCC-TZ = AE RI-MP2/aug-cc-pVTZ
       + FC CCSD/aug-cc-pVDZ
       - FC RI-MP2/aug-cc-pVDZ
```

三个分量均包含轨道响应；没有拟合参数。冻结核数：H–He 0、Li–Ne 1、Na–Ar 5。
采用 PySCF 默认收敛设置，已移除 HF 稳定性分析，仍检查求解收敛、矩阵电子数/对称性和网格积分。
数值检查不保证任意分子的 η <1；历史独立验证的 C₃ 失败记录保留在验证归档。

闭壳层批处理保存 aTZ 的 MP2 密度与 aDZ 的 CCSD−MP2 修正两个 AO 矩阵，float64 下三角无损压缩。
不投影到共同基组、不截断占据数、不重新归一化电荷；实空间密度按需重建。
`build` 保存三个原始分量，`dataset.write_compact` 转为两矩阵格式。
开壳层保留 R/U 路径，可写原始自旋分量，尚未接入默认闭壳层压缩批处理。
`pyscf/mp/dfmp2_native.py` 中保留已验证的矩形 AO×MO 维数修复：MO 数取系数矩阵列数。
