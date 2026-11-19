# DCC-TZ（H3C）与小分子密度输入

方法在 `pyscf/dcc_tz.py`，可直接 `from pyscf import dcc_tz`，不依赖 `work/`。
工作区只有两个业务代码文件：`dataset.py` 处理输入、筛选与密度存储，`run.py` 处理运行与提交。

**生产输入共 4,192 个分子构型：3,913 个闭壳层、279 个开壳层。** 数据在 `data/small-density.zip`，
包含坐标、电荷、自旋、片段、来源、筛选报告、原始数据及许可证，可直接供批处理读取。
这些是待计算的输入，不是已有高精度密度标签，也不是新建的独立精度测试集。

## 数据选择

保留当前 IXC2 训练库的小分子基础池，再从 NCI Atlas、A24、O24×5 和 GMTKN55 补充相互作用及自旋态。
下表“构型数”包括需要计算的非原子单体，已按精确笛卡尔坐标、电荷、自旋去重。
不同取向、不同原子排序或仅近似相同的几何不合并；不等于不同化学式的数量。

| 来源 | 主要内容 | 所选复合物几何 | 默认计算构型数 |
| --- | --- | ---: | ---: |
| TAE25 | H–Ar 共价小分子、平衡构型 | — | 1,499 |
| BSE49 | 115 个闭壳层母体、60 个双重态自由基 | — | 175 |
| D442×10 | 色散、弱相互作用 | 41 条曲线 ×10 点 | 625 |
| HB375×10 | O/N/C–H 中性氢键及原集 noHB 对照 | 17×10 | 204 |
| IHB100×10 | 阳离子/阴离子参与的氢键 | 21×10 | 252 |
| HB300SPX×10 | 扩展到 S/P/F/Cl 的氢键 | 53×10 | 636 |
| SH250×10 | 卤键、硫族键、磷族键 | 9×10 | 108 |
| R739×5 | 短程排斥、不同元素接触 | 66×5 | 387 |
| A24 | 小分子偶极、混合相互作用、有机色散、非芳香 π 接触 | 24 个原始构型 | 70 |
| IL16 | 正负离子对：乙铵–氯离子；Cl⁻ 原子参考另列 | 1 个原始构型 | 2 |
| O24×5 | 高自旋非共价复合物、开壳层及闭壳层单体 | 22×5 | 146 |
| W4-11 | 中性小分子自由基、三重态 | — | 35 |
| G21EA | 中性及阴离子开壳层分子 | — | 27 |
| G21IP | 中性及阳离子开壳层分子 | — | 26 |
| **合计** | **229 条完整非共价曲线 +24 个 A24 构型 +1 个离子对** | **1,875 个复合物构型** | **4,192** |

总数为 2,239 个 IXC2 闭壳层基础构型、60 个 IXC2 自由基、88 个 GMTKN55 独立开壳层构型、
1,465 个新增复合物和 379 个新增单体记录，去除 39 个精确重复。
新增曲线的单体只取原始 scaling=1.0 几何，不优化、不加 ghost 原子；片段内部距离沿曲线的最大变化
见 `selection/curves.json`（逐条记录，保持源坐标）。不把三个密度分量计作三个样本。

筛选规则固定在 `dataset.prepare_small`：

- 元素限定 H–Ar；默认批次包含闭壳层与开壳层。电荷分布为 0：3,923，+1：71，−1：187，−2：11。
- TAE25/BSE49：≤3 个重原子且 aTZ ≤200 AO；D442：完整复合物 aTZ ≤300 AO。
- 新 NCI 集首先保留 aTZ ≤300 AO 的完整曲线。对仍缺失的类别，最多补一个最小代表，且 ≤400 AO。
  补充为 1 条 NH–O 中性氢键、4 条阳离子氢键、7 条 σ-hole 曲线；降低阈值会丢掉这些类别。
- R739 对每个 `(原始 group, 接触元素对)` 只保留最小的一个完整五点曲线，避免同类构型过量。
- IL16 的 16 个离子对中仅乙铵–氯离子在 400 AO 以内（372 AO），纳入一个原始构型及乙铵单体。
- A24 保留全部 24 个原始几何。其分组由本项目整理，不表示相互作用具有纯粹的单一物理成分。

实际 aTZ AO 数中位数 **188**，P95 **299**，最大 **395**；134 个构型超过 300 AO，占 3.20%。
最大 aDZ 为 192 AO，最多 13 个原子。这些是规模指标，不是总耗时预测；本轮未恢复计时任务。

覆盖限于上述小分子类型：没有芳香环堆积、过渡金属、一般反应过渡态或溶剂构象系综。
SH250 的 Cl–芳香碳类别因最小体系也有 514 个 aTZ AO 而未选入。
IHB 提供带电氢键，IL16 补充一个真正的正负离子对；离子对的化学多样性仍有限。
BSE49 提供母分子/自由基端点，不是连续断键路径。

另外备好 **80 个派生双原子键长扫描点**，不进入默认批次。
扫描母体为 H₂、LiH、HF、HCl、N₂、CO、F₂、Cl₂，按源键长的
0.8、0.9、1、1.1、1.25、1.5、1.75、2、2.5、3 倍生成，保留源母体 ID。
这是 singlet 分支的研究挑战输入；深度拉伸可能需要多参考方法，不能直接保证 H3C 精度。

## 开壳层覆盖与处理

279 个开壳层输入由 60 个 BSE49 自由基、88 个 W4-11/G21EA/G21IP 构型，以及 O24×5 的
110 个复合物和 21 个开壳层单体组成。O24 另带来 15 个闭壳层单体，因此闭壳层池从 3,898 增至 3,913。

| 多重度 2S+1 | 构型数 |
| --- | ---: |
| 双重态 | 196 |
| 三重态 | 68 |
| 四重态 | 5 |
| 五重态 | 10 |

其中中性 242 个、阳离子 26 个、阴离子 11 个。开壳层最多 11 个原子，aTZ 最大 345 AO。
独立开壳层分子沿用 ≤3 重原子、aTZ ≤200；O24 保留 0.9、1.0、1.2、1.5、2.0 五个距离点。
其唯一超过 300 AO 的曲线是乙烯基自由基–乙烯（345 AO），补充自由基–烯烃接触。
原 O24 分类覆盖 9 个色散主导、6 个静电主导、7 个混合作用复合物。

`spin` 明确表示 Nα−Nβ；O24 两个片段取 S_total=S_A+S_B 的高自旋耦合，绝不把两个开壳层片段自动写成 singlet。
原集的 Na–Li 高自旋曲线因当前冻结核会移除全部活性 β 占据轨道而排除；He*–He* 亚稳激发原子对也排除。
未改变冻结核规则来强行纳入它们。低自旋自由基对、broken-symmetry singlet 及激发态跟踪不在本批范围。
O24 使用原始无 midbond 几何，不增加基函数中心。其原论文 ROCCSD(T) 能量不能充当本方法 UHF 密度的精度验证。

开壳层沿用相同 H3C 三项组合，分别用 UHF、RI-UMP2、UCCSD；两种基组均保留 α/β 通道。
检查每个通道的 AO 电子数、复合密度的网格 Nα/Nβ 和 Nα−Nβ，并记录 UHF ⟨S²⟩；不恢复稳定性分析。

## 来源与可追溯性

还检查了 GMTKN55 的 IL16 集。其余 15 个离子对为 464–694 个 aTZ AO，超出本批预算；
报告记录全部 16 个体系的大小，包内保留用于核对的原始几何。

- [TAE25 原始数据](https://huggingface.co/datasets/microsoft/msr-acc-tae25)：CDLA-Permissive-2.0；[原论文](https://arxiv.org/abs/2506.14492)。
- [GMTKN55 作者仓库](https://github.com/grimme-lab/GMTKN55)：CC BY 4.0，固定 revision `8d485b37a1ca8837e395042671ca5ba4e0714691`；
  [GMTKN55 原论文](https://doi.org/10.1039/C7CP04913G)、[IL16 原论文](https://doi.org/10.1021/ct5010593)。
  开壳层源 ZIP 只提取 W4-11/G21EA/G21IP 原始 XYZ、`.CHRG`、`.UHF`、文献和许可证；保留原文件字节及上游 tarball 校验值。
- [BSE49 原论文](https://www.nature.com/articles/s41597-021-01088-2)。
- [NCI Atlas 作者仓库](https://github.com/Honza-R/NCIAtlas)：CC BY 4.0；固定 revision `1816bfc72609d7deb1d4f93ab9e27eb13bb44bec`。
  原 README 列出每个子集的作者和论文，完整保存在输入 ZIP 中。
- [O24×5 原论文](https://doi.org/10.1063/5.0043793)及[作者仓库](https://gitlab.com/piotr.zuchowski/o24x5)；几何取自同一固定 Psi4 revision 的 `O24by5.py`，保留 LGPL-3.0 原模块。
- [A24 原论文](https://doi.org/10.1021/ct400057w)；几何取自 [Psi4 A24 模块](https://github.com/psi4/psi4/blob/c3cd35420200a5c7122930b517c339056e763c85/psi4/share/psi4/databases/A24.py)。
  保留未修改的模块及 LGPL-3.0 许可证；仅解析字面量，不执行下载的源码。

IXC2 基础池来自活动配置 `relative-df-20261114-quota-v686` 指定的训练库，数据库 SHA256：
`a0a9c59bbb09b2aa13ef0feee1a3c69439b7d53e48ef69682f9767376c32542c`。
基础闭壳层池没有与该库 held-out 观测共享的 state ID；27 个共享的开壳层态单列在原筛选归档。
新增公开数据尚未分配 train/test，且可能与其他基准有来源或几何重叠。
将来分割训练和验证时，应连同曲线、相关单体一起分组，再检查跨来源重叠。

`small-density.zip` 内：

- `cases/production.json` 与 `dcc/inputs/*.jsonl`：同一批 4,192 个输入，后者预分成 256 个调度分片。
- `cases/closed_shell.json`、`cases/open_shell.json`：默认池的两个互不重叠子集，可分别提交。
- `dcc/closed-shell/`、`dcc/open-shell/`：子集调度清单，分别为 256、16 个分片；不是机器学习 train/test 分割。
- `cases/bond_stretch_challenge.json`：默认队列之外的断键挑战输入。
- `cases/atomic_references.json`：原子参考记录，另计，不在分子计算数中。
- `report.json`：选取数量、缺失类别、大小分布、来源校验值及规则。
- `selection/curves.json`、`selection/exact_aliases.json`：完整曲线与重复记录映射。
- `sources/`：未修改的公开源文件，以及原 `ixc2-small.zip` 的筛选、能量观测、许可证和来源快照。
- `ATTRIBUTION.txt`、`index.json`：出处说明及逐文件 SHA256。

其他两个 ZIP 为 `qm9.zip`（备用输入）和 `validation.zip`（历史验证与已停止计时），均不推送到 Git。
原独立 `ixc2-small.zip` 完整收进新输入包，避免重复保留。

每个生产输入与密度标签的 `source_datasets` 明确标注原始数据集；`provenance` 保留源归档成员、
SHA256、构型编号或原始能量观测 ID、IXC2 state ID 及中间数据库校验值。
IXC2 基础记录明确归为 TAE25、BSE49 或 D442×10，不再仅以 IXC2 作为来源名。
精确重复的构型合并出处；派生单体标记片段提取及母体编号。每个输出 ZIP 的索引也记录原始数据集名称。

## 使用

在仓库根目录激活环境。以下命令只准备配置和预览，不提交计算：

```bash
source .local/activate.sh
export PYTHONDONTWRITEBYTECODE=1
python -m work.run prepare
python -m work.run submit --concurrency 4
```

默认输入 `work/data/small-density.zip`，默认输出 `work/runs/small-density`。
256 个分片按大小分为两组，每组内按 AO 数交错分配；每个作业内依次计算多个分子并保存检查点。
每个分子严格使用一个节点、一个进程，`srun` 绑定分配的 CPU 核；不独占节点。

| 资源组 | 构型数 | 分片数 | 每任务物理核数 | 请求内存 | 求解器内存上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| aTZ ≤200 AO | 2,921 | 128 | 4 | 8 GiB | 6,000 MB |
| aTZ >200 AO | 1,271 | 128 | 8 | 16 GiB | 12,000 MB |

核数是面向批量吞吐的初始配置，未声称已证明最优。无 `--concurrency` 时不设额外节点并发限制，
由 CPU 分区、用户 `cpu=1000` 的 QoS 限额与现有占用决定实际并发，其余任务排队。
该集群按 core 分配、按逻辑 CPU 记账；4/8 个物理核通常分别记作 8/16 个配额 CPU。
不占用 GPU 分区，不修改或取消其他工作负载。当前分区不按内存资源调度，因此每次开始分子前检查实际可用 RAM。

实际提交与查看进度：

```bash
python -m work.run submit --execute
python -m work.run report
```

提交会跳过本运行已经在排队或执行的分片；`--concurrency N` 可选，仅限制本次新提交数组的并发任务总数。
每天到时前保存已完成标签并自动重新排队；手动取消不会自动恢复。
单个分子最多 8 小时，求解失败/超时单独记录，需要显式 `--retry-failed` 重试，不混入有效标签。
积分临时文件仅写节点本地 `/tmp`，本地临时盘保留至少 10 GiB，共享输出保留至少 100 GiB；
cpu05/cpu06 的本地 `/tmp` 为约 94 GiB 的内存盘，其他已检查节点为磁盘；始终检查实际可用 RAM。
内存或磁盘不足时保存检查点并重新排队。
批次配置和每次提交的完整参数保存在运行目录中；代码及输入校验值锁定后再开始运行。
单独准备开壳层并预览（仍不提交）：

```bash
python -m work.run prepare --subset open-shell --root work/runs/open-shell
python -m work.run submit --root work/runs/open-shell --concurrency 4
```

单独闭壳层用 `--subset closed-shell`。配置固定输入 SHA256、子集、方法和流程源码；修改源码/输入后应使用新运行目录。
本次生产的节点临时盘阈值修复单独记录在 `runtime-policy.json`：必须精确匹配新旧源码校验值及运行 ID，
只允许调整临时盘余量，不允许改变方法、输入、数据处理代码或数值设置；保留已完成密度及其运行身份。
修复后的标签还记录实际执行的流程源码校验值。初次绑核诊断的 7 份标签及日志已归入
`validation.zip/production-startup-20261119/`，正式运行仍在默认输出目录。

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

闭壳层保存 aTZ 的 MP2 密度与 aDZ 的 CCSD−MP2 修正两个 AO 矩阵。
开壳层保存 **aTZ 的 Dα、Dβ，以及 aDZ 的 ΔDα、ΔDβ，共四个 AO 矩阵**；按两个基组块组织，
每块下三角数组形状为 `(2, nAO*(nAO+1)/2)`。均使用 float64 无损压缩。
新格式 `dcc-tz-spin-v2` 同时支持两类，读取器兼容旧闭壳层 v1。批处理逐位核对两个自旋通道并支持分片恢复。
不投影到共同基组、不截断占据数、不重新归一化电荷；实空间密度按需重建。
`build` 保存三个原始分量，`dataset.write_compact` 转为上述紧凑格式。
`evaluate_compact(..., spin=True)` 返回 `(ρα, ρβ)`，默认返回总电子密度；磁化密度为 ρα−ρβ。
不会把开壳层的自旋密度按总密度的一半重建。
`pyscf/mp/dfmp2_native.py` 中保留已验证的矩形 AO×MO 维数修复：MO 数取系数矩阵列数。

## 本次验证

2026-11-19 在单节点单线程作业 1056974 中，OH 双重态与 O₂ 三重态完成实际 H3C 计算、
分自旋网格积分、逐位压缩往返、实空间重建及分片写入/恢复。积分 Nα/Nβ 分别接近 (5,4)、(9,7)，
每通道误差均小于 5×10⁻⁹ 个电子。另有 8 项格式/入口检查通过，包括旧闭壳层兼容、错误 α/β 交换、缺失 β、NaN、错误自旋/运行身份的拒收。
279 个开壳层及新增闭壳层单体共通过 588 次基组/电子数/冻结核输入检查；全部 22 条 O24 曲线完整。
这些检查验证实现与存储，不建立开壳层 η<1 的精度结论。实时生产进度以 `work.run report` 为准。
检查脚本、日志和两个原始/压缩密度收进 `validation.zip/open-shell-support-20261119/`。
