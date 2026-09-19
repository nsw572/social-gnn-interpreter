# Social-GNN Interpreter：理想架构与项目状态报告

更新日期：2026-09-19

## 1. 项目定位

本项目是 `deepOF/social_gnn` 的下游解释层。它接收已经完成视频特征提取、时间对齐和 trial 级 padding 的 Social-GNN batch，调用冻结的 `SocialGNNWithTCN` 获得时序表征 `s_t`，再完成聚类对比、干预、边界搜索和报告生成。

本项目不负责：

- 读取原始视频、H ZIP 或原始 CSV；
- 重复实现 node/edge 特征提取；
- 重新建立 social clock；
- 跨 trial 拼接时间序列；
- 把表示空间反事实误称为真实行为因果效应。

上游当前 A/B latent 完全重复的数据被认定为 upstream recombination failure，只能用于接口联调，不得用于训练 encoder、decoder、scaler 或 clusterer。

## 2. 理想的端到端架构

```text
deepOF 上游（本项目不重复实现）
  原始视频 / H 数据
    -> H ZIP -> 标准 node NPZ
    -> 视频 -> 8 维 edge value + 8 维 confidence
    -> 权威 patch clock
    -> SocialTrialPackage / trial-level DataLoader

稳定的数据合同
  node_features     [B,T,N,D_node]
  node_mask         [B,T,N]
  edge_values       [B,T,N,N,8]
  edge_confidence   [B,T,N,N,8]
  time_mask         [B,T]
  patch_start_s     [B,T]
  patch_end_s       [B,T]
  trial_id          [B]
  training_eligible [B]
    |
    v
冻结的 SocialGNNWithTCN
  compose_edge_inputs -> 16 维 edge 输入
  Graph Encoder（逐 social timestep）
  causal TCN
  -> s_t [B,T,D_s] + time_mask + provenance
    |
    +-------------------------------+
    |                               |
    v                               v
SSL / Readout                    Interpreter
  target builder                  1. Contrast
  temporal decoder                  rival cluster
  masked objective                  trial-aware statistics
  readable behavior heads           candidate factors
    |                               2. Perturbation
    |                                 latent sensitivity
    |                                 canonical behavior edit
    |                               3. Boundary
    |                                 constrained search
    |                                 minimal counterfactual
    +---------------+---------------+
                    v
           validation + reporting
           JSON / text / audit trail
```

### 2.1 推荐的模块边界

```text
social-gnn-interpreter/
  interpreter/
    core.py                 # 不可变 trial sample、provenance、适配器合同
    social_gnn.py           # 内存 batch -> s_t；canonical rerun
    clustering.py           # 聚类 artifact、scaler/pipeline 和版本校验
    contrast.py             # rival 选择和 trial-aware 统计
    perturbation.py         # 干预协议和批量敏感性实验
    boundary.py             # 受约束的边界搜索
    validation.py           # 结构、经验支持和行为合理性约束
    ssl.py                  # decoder / target / masked loss 接口
    readouts/               # 未来的可读行为 decoder
    interventions/          # 未来的 canonical 行为干预
    report.py               # 机器可读和人类可读报告
    pipeline.py             # 组合层，不读取原始数据
  tests/
    unit/                   # 数值、mask、provenance、约束
    integration/            # 与真实 deepOF API 的兼容性
    synthetic/              # 几何一致性和可恢复性场景
  configs/                  # 未来的 target、artifact 和评估配置
  artifacts/               # 仅保存 manifest；大模型文件不进入 Git
  PROJECT_STATUS.md
  README.md
  pyproject.toml
```

`clustering.py`、`readouts/`、`interventions/`、`configs/` 和 artifact manifest 是理想架构中的后续模块，目前尚未完整实现。

## 3. 当前已经完成的部分

### 3.1 Trial-aware 核心合同

- `BehaviorSample` 保存完整 trial 的 `s_t`、`time_mask`、patch 时钟和 metadata。
- 所有数组采用防御性复制和只读语义。
- padding timestep 不进入聚类和统计。
- 每条有效观测保留 `(trial_id, timestep, patch_start_s, patch_end_s)` 来源。
- 数据单位保持为完整 trial，不跨 trial 拼接时间。

### 3.2 Social-GNN 对接

- `SocialGNNEncoderAdapter` 接受已加载的内存 batch，不读取 ZIP、NPZ、CSV 或视频。
- 复用上游 `social_gnn.graph_builder.compose_edge_inputs`。
- 检查 edge value 与 confidence 合并后为 16 维。
- 显式传递 `time_mask`，运行冻结模型并提取 `s_t`。
- 保留 trial id、权威 patch 时间和可用 metadata。
- `CanonicalReencodingRunner` 可以调用外部提供的 batch transform，再运行 encoder 和 clusterer。

### 3.3 三层 Interpreter MVP

- Contrast：最近 cluster representative rival 选择；逐 trial 中位数汇总；Cohen's d 候选排序。
- Perturbation：潜变量坐标插值；重新调用 cluster adapter；保存逐观测结果。
- Boundary：标准化 L2 代价；有界网格搜索与二分细化；生成最小表示空间反事实。
- 支持 sklearn KMeans 和 GaussianMixture 的 out-of-sample assignment/evidence。
- 输出结构化 JSON 和文本报告。

### 3.4 SSL 基础设施

- 已定义 `TemporalDecoder`、`PredictionTargetBuilder`、`PredictionBatch`、`TargetBatch` 和 `MaskedLoss` 接口。
- 已实现同时考虑 `time_mask` 与 element mask 的 `MaskedMSELoss`。
- target 会被 detach/clone，padding 不参与 loss。
- 尚未选择具体 SSL 任务，因此没有具体 decoder、target builder、训练配置或已训练权重。

### 3.5 测试

- 现有单元测试覆盖 sample 不可变性、mask、provenance、聚类适配器、contrast、perturbation、boundary、report 和 SSL mask。
- 可选 deepOF integration test 调用真实 `SocialTrialPackage`、collate 和 `SocialGNNWithTCN` API。
- integration smoke test 使用合成内存 batch；它证明接口相容，不证明真实数据效果或生物学有效性。

## 4. 当前未完成或仅有接口的部分

### 4.1 可读行为 decoder

当前默认候选是 `latent_0`、`latent_1` 等表示坐标。尚不存在经过训练和验证的 decoder，将 `s_t` 映射到鼠间距离、相对朝向、接近速度或其他可读行为变量。

在真实数据不存在时，可以确定 target schema、masked multi-head decoder 接口以及合成测试，但不能证明 `s_t` 包含目标行为信息，也不能产生可信的行为解释。

### 4.2 具体 SSL 任务

SSL 思路尚未确定。未来 target 可来自已经对齐的 `edge_values`、`node_features` 或未来时刻表征，但必须满足：

- decoder 的主要输入边界只有 `s_t`；
- target builder 可以在训练阶段读取对齐 batch，但不得修改原始 target；
- masking/augmentation 在加载后动态执行；
- loss 明确记录有效 timestep 和逐元素有效性；
- 任务选择由可验证的科学问题驱动，而不是为填充接口任意指定。

### 4.3 Canonical 行为干预

当前有 canonical rerun 的调用钩子，但没有具体、物理一致的行为编辑器。真正的 canonical intervention 应修改 node-level 几何或轨迹，并由上游权威函数重新计算所有耦合的 edge value、confidence 和 mask；不能只改一个“距离”数值而保留互相矛盾的方向、速度和接近特征。

在没有真实数据时，可以实现协议、合成几何场景和不变量测试；不能估计真实小鼠行为的经验支持范围，也不能宣称干预具有生物学合理性。

### 4.4 真实 artifact 与科学评估

目前缺少：

- 通过有效数据训练并冻结的 Social-GNN checkpoint；
- 与 encoder 严格配套的 scaler 和 KMeans/GMM artifact；
- decoder checkpoint 与 target manifest；
- 真实 trial 级 train/validation/test split；
- 稳定性、可重复性、bootstrap 区间和外部行为效度；
- 从输入 artifact 到报告的版本化审计链。

## 5. 已知上游风险

1. converter 能标记 `training_eligible=false`，但该标志目前未完整贯穿 `SocialTrialPackage -> torch sample -> collate`。训练 manifest 必须显式拒绝重复 latent trial。
2. 部分上游 KMeans/GMM helper 在 fit 时未使用 `time_mask`。若 artifact 已被 padding 污染，下游无法补救。
3. directed-edge 聚合方向、invalid node 消息屏蔽和对角 self-edge 语义仍需要专门测试。
4. clustering artifact 必须包含与训练一致的预处理/scaler；只保存裸 estimator 可能造成推理尺度不一致。

## 6. 完成度判断

| 范围 | 估计完成度 | 当前结论 |
|---|---:|---|
| Social-GNN 接口与 trial/mask/provenance | 85–90% | 基本完成，待真实 artifact 契约验证 |
| 表示空间 Interpreter 闭环 | 80–85% | 可运行的 MVP |
| SSL | 约 50% | 接口和 masked loss 完成，任务与模型未定 |
| 可读行为解释 | 30–40% | 有 feature/decoder 边界，无有效 decoder |
| Canonical 行为干预 | 25–35% | 有 rerun hook，无具体一致性编辑器 |
| 真实数据科学验证 | 10–20% | 只有接口 smoke test |
| 整体目标 | 约 60% | 软件骨架成立，科学系统尚未完成 |

“25 项测试通过”只能支持接口和数值实现的工程结论，不能等同于真实行为解释已经成立。

## 7. 推荐推进顺序

### 阶段 A：在无真实数据情况下可完成

1. 固化 batch schema、feature index、单位、方向约定和 `training_eligible` 传播规则。
2. 为 encoder、scaler、clusterer、decoder 定义 versioned artifact manifest。
3. 将 SSL 保持为策略接口，先确定科学问题，再选择 target。
4. 定义可读行为 target registry，但暂不声称 decoder 有效。
5. 定义 canonical intervention protocol，并用合成二维轨迹测试几何一致性、mask 和时间连续性。
6. 增加 CI，分别运行纯 NumPy 测试和可选 deepOF integration 测试。

### 阶段 B：真实有效数据到位后才能完成

1. 建立 trial-level split，并排除所有 `training_eligible=false` 数据。
2. 训练和冻结 encoder、scaler、clusterer，并验证 padding 不进入 fit。
3. 比较候选 SSL/readout 任务的泛化性能和可校准性。
4. 用真实分布拟合干预支持约束，拒绝 out-of-support counterfactual。
5. 对 cluster 稳定性、干预响应和边界距离进行 bootstrap/重复训练评估。
6. 只有在 decoder 精度和 canonical intervention 有效性通过预设标准后，才输出可读行为结论。

## 8. 项目当前准确标签

> 一个与 Social-GNN 正确解耦、保留完整 trial/mask/provenance、能够执行表示空间 Contrast -> Perturbation -> Boundary 的下游 Interpreter MVP。SSL 和 canonical re-encoding 已有扩展接口；可读行为 decoder、具体 SSL 任务、物理一致的 canonical 行为干预以及真实数据验证尚未完成。
