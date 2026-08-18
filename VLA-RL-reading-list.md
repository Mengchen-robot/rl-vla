# VLA + RL：从快速入门到当前研究版图

- 更新至：2026-08-17
- 定位：入门 **VLA + RL post-training**，不是单纯入门 VLA 或 RL
- 未列入：OpenVLA、RT-2、Octo、π0 等纯 VLA 基础论文
- 论文状态与版本以链接页面为准

主问题不是“哪个 RL 算法最好”，而是：

1. 动作分布能否算 likelihood
2. 奖励从哪里来
3. rollout 在哪里发生
4. 如何避免真机损耗
5. 更新后是否保留 generalist 能力

## 先记住这三个判断

1. 目前 VLA+RL 仍以 **post-training** 为主，不是从零用 RL 训练通用 VLA。
2. PPO、GRPO 谁更好取决于动作参数化、同初态采样、奖励密度与系统吞吐，**不能照搬 LLM-RL 结论**。
3. 多数论文的数字不可横向比较。LIBERO / RoboTwin 增益也不等价于真机开放世界泛化。

## 阅读路线

顺序比年份重要。建议先完成主线，再按研究问题走分支。

| 路线 | 阅读顺序 | 用来干什么 |
| --- | --- | --- |
| 最短主线 · 8 篇 | Survey → iRe-VLA → ConRFT → RL4VLA → RIPT-VLA → VLA-RL → πRL → SOP | 两三天内建立全局坐标：稳定化、offline-to-online、PPO/GRPO、reward、flow policy、真机系统 |
| 真机部署线 · 7 篇 | ConRFT → iRe-VLA → SOP → GR-RL → TwinRL → IG-RFT → BORA | 比较人工接管、replay、reset、residual、数字孪生、接触状态与多机并行 |
| 算法与规模线 · 7 篇 | RL4VLA → RIPT-VLA → VLA-RL → SimpleVLA-RL → RLinf-VLA → πRL → Temporal GRPO | 比较 critic vs value-free、PPO vs GRPO、稀疏 vs 过程奖励、自回归 vs flow、轨迹级 vs 阶段级 credit |
| 低真机交互线 · 5 篇 | ConRFT → World-VLA-Loop → WoVR → TwinRL → BORA | 比较 offline critic、视频世界模型、显式 twin 与 residual adapter 的成本—可靠性边界 |

## 七种主流训练范式

真正需要比较的是范式，而不是单篇榜单。

| 范式 | 典型 recipe | 优势 | 硬伤 | 代表 |
| --- | --- | --- | --- | --- |
| 仿真 on-policy | SFT warm start → PPO / GRPO / RLOO rollout → success/process reward | 并行度高；能主动探索；适合比较算法和扩规模 | sim gap；稀疏奖励；长轨迹信用分配；benchmark 容易饱和 | RL4VLA, RIPT, VLA-RL, SimpleVLA-RL |
| 真机 HIL off-policy | demo / offline critic → replay buffer → human intervention → online actor update | 样本效率高；安全；可 reset-free；适合 contact-rich specialist | 人力、奖励、重置、系统工程重；跨任务泛化通常有限 | ConRFT, IG-RFT |
| RL ↔ SFT 交替 | 冻结大 backbone 做 RL 探索 → 收成功轨迹 → 全模型 SFT | 稳定、便宜；成功经验可固化；降低大模型在线更新风险 | 不是完全 on-policy；会丢失失败信息；能力受 action head 探索限制 | iRe-VLA |
| Residual / adapter RL | 冻结 VLA prior → 学 action residual 或 latent noise correction | 防遗忘、低硬件风险、模型无关、适合高维精细控制 | 修正空间受 base policy 限制；很难获得全局新技能 | GR-RL, BORA |
| World model / digital twin | 学习视频 simulator 或重建 twin → imagined RL → 少量真机闭环校准 | 大幅减少真机 rollout；可并行探索困难状态 | 模型漏洞、长时序漂移、物理不一致；可能优化出“骗 simulator”的策略 | World-VLA-Loop, WoVR, TwinRL |
| Preference / constrained RL | 干预 desirability 或 reward-cost 约束 → 对齐更新 | 反馈语义更丰富；可显式平衡成功与安全 | 反馈成本与 specification gap；偏好优化不等于环境 RL | APO, SafeVLA |
| Fleet online system | 多机器人异步执行 → 云端 learner → 频繁同步新 policy | 提升 on-policy 新鲜度与墙钟效率；支持多任务共享策略 | 硬件和运维门槛最高；系统、数据和算法难拆分 | SOP |

## 论文清单

每篇只抓 pipeline、假设、证据边界和失败模式。

### 主线

#### 1. [A Survey on Reinforcement Learning of VLA Models for Robotic Manipulation](https://doi.org/10.36227/techrxiv.176531955.54563920/v1)

- 年份：2025
- 流派：总览 / taxonomy
- 训练：online、offline、test-time；部署与评测
- 模型：跨架构
- 任务：综述
- 为什么读：先建立 action / reward / transition、online / offline / test-time、sim-to-real / HIL 的统一坐标系。
- 优点：覆盖面最贴合 VLA+RL；适合做索引。
- 局限：是预印本综述；对快速变化的新论文与实验可信度需自行复核。

#### 2. [Improving Vision-Language-Action Model with Online Reinforcement Learning (iRe-VLA)](https://arxiv.org/abs/2501.16664)

- 年份：2025 · ICRA
- 流派：RL ↔ SFT 交替
- 训练：在线 RL 只训 action head；成功轨迹再全模型 SFT
- 模型：VLM + 轻量 action head
- 任务：仿真 + Panda 真机抓取
- 为什么读：理解最朴素也最实用的稳定化思路：RL 负责探索，SFT 负责把成功经验写回大模型。
- 优点：算力压力小、训练稳定、真机验证清楚。
- 局限：RL 阶段冻结 backbone，探索能力仍受表征和初始策略支撑限制。

#### 3. [ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy](https://arxiv.org/abs/2502.05450)

- 年份：2025 · RSS
- 流派：offline → HIL online RL
- 训练：BC + Cal-QL 初始化；人类干预在线强化
- 模型：Octo + consistency action head
- 任务：8 个 Franka 真机任务；接触丰富
- 为什么读：真机派代表作：把离线 critic、在线 replay、人工接管、安全探索接成完整系统。
- 优点：45–90 分钟真机适配；样本效率高；适合工业 specialist。
- 局限：依赖任务奖励、重置和人工干预；更像 VLA prior 上的 task-specific RL。

#### 4. [What Can RL Bring to VLA Generalization? An Empirical Study](https://arxiv.org/abs/2505.19789)

- 年份：2025 · NeurIPS
- 流派：算法与泛化对照
- 训练：PPO vs GRPO vs DPO；RL vs SFT
- 模型：OpenVLA + LoRA
- 任务：pick-and-place；视觉 / 语义 / 执行 OOD
- 为什么读：先读结论再读 GRPO 热潮：其设置中 PPO 更稳；RL 主要改善 execution，其次 semantics，对纯视觉 OOD 未明显优于 SFT。
- 优点：控制变量和诊断维度清楚，是判断“RL 到底带来什么”的基准。
- 局限：任务族较窄；不能据此断言所有架构、奖励和长时序任务都应选 PPO。

#### 5. [Interactive Post-Training for Vision-Language-Action Models (RIPT-VLA)](https://arxiv.org/abs/2505.17016)

- 年份：2025
- 流派：value-free on-policy
- 训练：稀疏二值奖励；RLOO + PPO；动态过滤无信息 rollout group
- 模型：QueST / OpenVLA-OFT
- 任务：仿真交互后训练
- 为什么读：理解不训练 critic 时，如何利用同 context 多 rollout 做相对优势估计。
- 优点：简单、跨模型、少演示；对稀疏成功奖励友好。
- 局限：同初态多次采样代价高；全成或全败的 context 无梯度；主要证据仍在仿真。

#### 6. [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable RL](https://arxiv.org/abs/2505.18719)

- 年份：2025
- 流派：process reward + scalable online RL
- 训练：轨迹级 RL；VLM process reward；课程与 critic warmup
- 模型：自回归 OpenVLA-7B
- 任务：LIBERO 40 tasks；test-time optimization
- 为什么读：看 reward engineering 如何从终局成功扩展到自动切段的过程奖励，以及系统并行如何决定 RL 是否跑得动。
- 优点：多任务规模较大；把奖励、课程、环境并行和训练稳定性放在一起。
- 局限：过程奖励会引入 reward-model 偏差；仿真结果不能直接外推到真机。

#### 7. [πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models](https://arxiv.org/abs/2510.25889)

- 年份：2025
- 流派：flow policy 的 on-policy RL
- 训练：为 flow matching action chunk 构造可优化的 policy objective
- 模型：π 系 flow-based VLA
- 任务：仿真 + 真机
- 为什么读：这是动作分布分水岭：自回归 token policy 的 PPO/GRPO 不能原样套到 flow policy。
- 优点：补齐 flow-based VLA 的在线 RL；面向高频、灵巧 action chunk。
- 局限：似然 / 信用分配更复杂，训练成本高；实现与特定 flow 参数化强相关。

#### 8. [SOP: A Scalable Online Post-Training System for VLA Models](https://arxiv.org/abs/2601.03044)

- 年份：2026
- 流派：fleet-scale 真机闭环
- 训练：10 台机器人异步采集；集中 learner；HG-DAgger / RECAP
- 模型：算法无关的 VLA post-training system
- 任务：叠衣、装箱、补货；双臂真机
- 为什么读：主线最后读系统：真正的瓶颈常不是 PPO 公式，而是 on-policy 新鲜度、异步同步、人工干预与 fleet utilization。
- 优点：多机器人、多任务、数小时级真机在线提升；部署视角最完整。
- 局限：基础设施门槛高；系统增益与算法增益难完全分离。

### 分支

#### 9. [SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](https://arxiv.org/abs/2509.09674)

- 年份：2025
- 流派：GRPO 式规模化
- 训练：并行 rollout / render；探索增强；VLA 专用 loss pipeline
- 模型：OpenVLA-OFT
- 任务：LIBERO / RoboTwin + 少量真机
- 为什么读：看 LLM-RL 工程栈如何移植到 VLA，以及 RL 发现演示外动作模式的案例。
- 优点：吞吐和可扩展性强；长时序仿真表现突出。
- 局限：与 RL4VLA 的 PPO>GRPO 结论并不矛盾：架构、奖励、并行采样和探索策略不同；需避免只看榜单。

#### 10. [RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training](https://arxiv.org/abs/2510.06710)

- 年份：2025
- 流派：训练基础设施
- 训练：PPO / GRPO；render-inference-training 混合资源调度
- 模型：OpenVLA / OpenVLA-OFT
- 任务：LIBERO / ManiSkill / RoboTwin
- 为什么读：准备复现时读：统一环境和架构接口后，才能公平观察算法差异与系统瓶颈。
- 优点：工程可复用；跨 simulator；报告 PPO 更稳定。
- 局限：主要价值是系统而非新 RL 原理；结果仍高度依赖 benchmark pipeline。

#### 11. [SafeVLA: Towards Safety Alignment of VLA via Constrained Learning](https://arxiv.org/abs/2503.03480)

- 年份：2025 · NeurIPS Spotlight
- 流派：safe RL / CMDP
- 训练：任务 reward + safety cost；min-max / constrained optimization
- 模型：VLA + 安全约束
- 任务：Safety-CHORES；长时序移动操作仿真
- 为什么读：把“成功率最大化”改成“约束内最大化”，理解 reward 与 cost 分离及长尾风险评测。
- 优点：安全问题形式化，提供专门 benchmark 与 OOD 风险测试。
- 局限：仿真约束不等于物理安全保证；cost specification 仍可能漏掉未知风险。

#### 12. [Human-assisted Robotic Policy Refinement via Action Preference Optimization](https://arxiv.org/abs/2506.07127)

- 年份：2025 · NeurIPS
- 流派：偏好优化 / 人类反馈
- 训练：干预轨迹；二值 desirability；自适应 action reweighting
- 模型：自回归 VLA
- 任务：仿真 + 真机操作
- 为什么读：它不属于严格在线 policy-gradient RL，但代表部署中更稳定的数据复用路线：从失败动作和纠正动作学习。
- 优点：无需同初态成对偏好；适合不可逆真机交互；比直接 DPO 更贴合 action token。
- 局限：探索弱于 RL；依赖人类接管质量；优化目标与最终任务回报并非完全一致。

#### 13. [GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2512.01801)

- 年份：2025
- 流派：generalist → dexterous specialist
- 训练：offline Q 作 progress / 数据过滤；对称增强；在线 latent residual RL
- 模型：VLA base + latent noise predictor
- 任务：真机穿鞋带等毫米级长时序任务
- 为什么读：看 RL 如何不是追求更 general，而是把 generalist prior 专门化为高精度技能。
- 优点：任务难、真机说服力强；展示次优人类演示可被 RL 超越。
- 局限：管线复杂且任务工程重；跨任务泛化不是主要目标。

### 前沿

#### 14. [World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy](https://arxiv.org/abs/2602.06508)

- 年份：2026
- 流派：video world model RL
- 训练：状态感知视频 simulator + reward；policy / world model 共演化
- 模型：OpenVLA-OFT + GRPO
- 任务：仿真 + 低量真机校正
- 为什么读：世界模型路线代表：用 near-success 数据改善 action-following，并让 policy failure 反哺 simulator。
- 优点：减少物理交互；显式处理 policy shift 后 simulator 失配。
- 局限：world-model exploitation 与长时序视觉漂移仍是根本风险。

#### 15. [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/abs/2602.13977)

- 年份：2026
- 流派：可靠 imagined RL
- 训练：可控视频模型；keyframe-initialized rollout；policy-model co-evolution
- 模型：VLA + action-conditioned world model
- 任务：LIBERO + 真机迁移评估
- 为什么读：与 World-VLA-Loop 对读：核心不是画面逼真，而是限制误差深度、防止 policy 利用模型漏洞。
- 优点：直接针对 hallucination、累积误差和 simulator exploitation。
- 局限：仍依赖真实数据覆盖；虚拟提升到真机提升之间缺少普适保证。

#### 16. [TwinRL-VLA: Digital Twin-Driven RL for Real-World Robotic Manipulation](https://arxiv.org/abs/2602.09023)

- 年份：2026
- 流派：digital twin → targeted real RL
- 训练：手机重建 twin；扩展 SFT support；仿真找 failure region；HIL 真机补点
- 模型：VLA + digital twin
- 任务：4 个真机操作任务
- 为什么读：与视频 world model 对照：显式数字孪生物理更可控，但每个场景重建与校准成本更高。
- 优点：并行探索后只把高信息状态送到真机；约 20 分钟级适配。
- 局限：twin fidelity 与场景资产是瓶颈；规模化到开放环境并不轻松。

#### 17. [IG-RFT: Interaction-Guided RL for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2602.20715)

- 年份：2026
- 流派：SFT → offline RL → HIL RL
- 训练：interaction-guided AWR；轨迹 + 子任务 dense reward
- 模型：flow-based VLA
- 任务：4 个长时序真机任务
- 为什么读：三阶段 offline-to-online 真机范式：按接触状态调探索强度，适合长时序、接触敏感任务。
- 优点：兼顾初始稳定、离线数据利用与在线纠偏。
- 局限：奖励与阶段设计较重；多阶段收益很难归因到单一组件。

#### 18. [BORA: Bridging Offline RL and Online Residual Adaptation for Dexterous VLA](https://arxiv.org/abs/2605.30226)

- 年份：2026
- 流派：offline critic → online residual
- 训练：认知 token + action chunk critic；冻结 VLA；HIL chunk residual
- 模型：dexterous VLA + residual adapter
- 任务：5 个灵巧手真机任务
- 为什么读：看高维灵巧控制下最保守的部署解：保留 base prior，只让残差修正物理执行误差。
- 优点：硬件风险低、抗遗忘、online 参数少；适合 plug-in adaptation。
- 局限：残差受 base policy 能力上限约束；critic 与干预信号仍是 task-specific。

#### 19. [Temporal GRPO: Beyond Trajectory-Level Credit in VLA Reinforcement Learning](https://arxiv.org/abs/2608.13026)

- 年份：2026 · Aug
- 流派：时序 credit assignment
- 训练：可检测 task stages；同阶段 rollout 比较；分段 advantage
- 模型：GRPO-style VLA
- 任务：RoboTwin 2.0 / LIBERO-Long
- 为什么读：截至 2026-08 的最新 credit-assignment 分支：避免末段失败把前面正确动作全部负强化。
- 优点：对长时序稀疏 outcome reward 的问题定义直接、可解释。
- 局限：依赖可可靠生成并检测的阶段；目前主要是仿真证据，尚不应视为稳定主流。

## 速查表

| # | 优先级 | 论文 | 年份 | 流派 | 模型 | 任务 / 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 主线 | [Survey](https://doi.org/10.36227/techrxiv.176531955.54563920/v1) | 2025 | taxonomy | 跨架构 | 综述 |
| 2 | 主线 | [iRe-VLA](https://arxiv.org/abs/2501.16664) | 2025 · ICRA | RL ↔ SFT | VLM + action head | 仿真 + Panda 真机 |
| 3 | 主线 | [ConRFT](https://arxiv.org/abs/2502.05450) | 2025 · RSS | offline → HIL | Octo + consistency | 8 个 Franka 真机任务 |
| 4 | 主线 | [RL4VLA](https://arxiv.org/abs/2505.19789) | 2025 · NeurIPS | PPO/GRPO/DPO | OpenVLA + LoRA | pick-and-place OOD |
| 5 | 主线 | [RIPT-VLA](https://arxiv.org/abs/2505.17016) | 2025 | RLOO | QueST / OpenVLA-OFT | 仿真 |
| 6 | 主线 | [VLA-RL](https://arxiv.org/abs/2505.18719) | 2025 | process reward | OpenVLA-7B | LIBERO |
| 7 | 主线 | [πRL](https://arxiv.org/abs/2510.25889) | 2025 | flow RL | π 系 flow VLA | 仿真 + 真机 |
| 8 | 主线 | [SOP](https://arxiv.org/abs/2601.03044) | 2026 | fleet system | 算法无关 | 双臂真机多任务 |
| 9 | 分支 | [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) | 2025 | GRPO 规模化 | OpenVLA-OFT | LIBERO / RoboTwin + 少量真机 |
| 10 | 分支 | [RLinf-VLA](https://arxiv.org/abs/2510.06710) | 2025 | 训练基建 | OpenVLA / OFT | LIBERO / ManiSkill / RoboTwin |
| 11 | 分支 | [SafeVLA](https://arxiv.org/abs/2503.03480) | 2025 · NeurIPS | CMDP | VLA + 安全约束 | Safety-CHORES 仿真 |
| 12 | 分支 | [APO](https://arxiv.org/abs/2506.07127) | 2025 · NeurIPS | 偏好优化 | 自回归 VLA | 仿真 + 真机 |
| 13 | 分支 | [GR-RL](https://arxiv.org/abs/2512.01801) | 2025 | specialist residual | VLA + latent noise | 真机穿鞋带等 |
| 14 | 前沿 | [World-VLA-Loop](https://arxiv.org/abs/2602.06508) | 2026 | video world model | OpenVLA-OFT + GRPO | 仿真 + 低量真机 |
| 15 | 前沿 | [WoVR](https://arxiv.org/abs/2602.13977) | 2026 | imagined RL | VLA + world model | LIBERO + 真机迁移 |
| 16 | 前沿 | [TwinRL](https://arxiv.org/abs/2602.09023) | 2026 | digital twin | VLA + twin | 4 个真机任务 |
| 17 | 前沿 | [IG-RFT](https://arxiv.org/abs/2602.20715) | 2026 | offline-to-online | flow-based VLA | 4 个长时序真机任务 |
| 18 | 前沿 | [BORA](https://arxiv.org/abs/2605.30226) | 2026 | residual adapter | dexterous VLA | 5 个灵巧手真机任务 |
| 19 | 前沿 | [Temporal GRPO](https://arxiv.org/abs/2608.13026) | 2026 · Aug | 时序 credit | GRPO-style VLA | RoboTwin 2.0 / LIBERO-Long |

## 读论文时统一记录这 8 项

1. Base VLA 与 action head：token / Gaussian / diffusion / flow / consistency？
2. 更新范围：全模型、LoRA、action head、residual，还是 latent noise？
3. 数据阶段：SFT、offline RL、online RL、再 SFT 的先后与比例？
4. rollout：仿真、world model、单机真机、fleet；是否同初态多采样？
5. reward：终局二值、阶段 dense、process RM、人工干预，还是 safety cost？
6. credit：critic / GAE、RLOO、group relative、trajectory 或 temporal stage？
7. 部署代价：真机分钟数、人工分钟数、reset 次数、并行 robot 数、GPU？
8. 保真性：原任务遗忘、OOD 维度、真机任务数、是否报告失败与置信区间？

## 当前版图的一句话结论

- **主流 recipe**：SFT warm start → 受约束探索 → replay / on-policy update → 用成功或干预数据稳住策略。
- **仿真算法派**：PPO 仍是稳健基线；GRPO/RLOO 省 critic，但依赖同 context 多 rollout 和有区分度的奖励。
- **真机派**：offline-to-online + HIL + residual/adapter 比“全模型纯在线 RL”更现实。
- **架构分裂**：自回归 VLA 易接 policy-gradient；flow/diffusion VLA 高频灵巧，但 objective 与 credit 更难。
- **下一竞争点**：可靠 world model、阶段级 credit、安全约束、fleet post-training，以及提升 specialist 时不损伤 generalist。

## 选取原则

直接研究 VLA 的 RL / reinforced post-training，或对该训练范式有关键诊断价值。未把 OpenVLA、RT-2、Octo、π0 等纯 VLA 基础论文列入主清单。
