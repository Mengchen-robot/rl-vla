# 目前 VLA+RL 真正值得做的方向

更新：2026-08-18  
前提：基于 [`VLA-RL-reading-list.md`](./VLA-RL-reading-list.md) 的范式地图，而不是单篇榜单。

## 先把不值得做的说清楚

下面几类已经很挤，除非有**新问题陈述**或**真机证据维度**，否则很难构成一篇有辨识度的工作：

1. 在 LIBERO / RoboTwin 上再做一个 PPO / GRPO / RLOO 变体，成功率再涨几个点。
2. 把 LLM-RL 配方原样接到 OpenVLA-OFT，再报一次 “RL > SFT”。
3. 再做一个 1–2 小时 HIL residual adapter，平均成功率 95%，但任务是桌面抓放。
4. 再训一个更漂亮的视频 world model，却不处理 **policy 利用模型漏洞**。
5. 只优化 token-level likelihood，却声称解决了 flow / diffusion VLA 的在线 RL。

当前阶段的主矛盾不是 “RL 能不能涨点”，而是：

> **RL 能把 generalist VLA 变成更好的 specialist，但几乎还没证明它能让 generalist 在部署中持续变强、且不变笨。**

## 一张总图：缺口按“问题”而不是按算法

| 优先级 | 方向 | 现在卡在哪 | 已有萌芽 | 还缺什么 |
| --- | --- | --- | --- | --- |
| P0 | RL 后训练的稳定性 / 反遗忘 | RL 改权重 = 改 prior；成功经验随 checkpoint 蒸发 | CRL-VLA, residual/adapter, SOP | 部署经验如何沉淀、跨任务复用、不伤 generalist |
| P0 | 可被 RL 信任的 transition | 想象 rollout 会 hallucination、累积误差、reward hacking | World-VLA-Loop, WoVR, TwinRL, VLA-RFT | 何时信模型、何时回真机；uncertainty / 校准 / 拒绝探索 |
| P0 | 长时序 + 生成式动作的 credit | 终局成败污染前面正确动作；flow 反传噪声大 | Temporal GRPO, πRL, BORA | 不依赖 oracle stage；chunk / denoising / 接触阶段统一 credit |
| P1 | 可迁移、难被 hack 的 reward | 稀疏成功不够；process RM 自己会偏 | VLA-RL, IG-RFT, SafeVLA | 物理接地（接触/力/进度）+ 语言约束，而不是更密的 shaping |
| P1 | flow/diffusion VLA 的原生 RL | 主流 RL 还按 token policy 想 | πRL, FPO, ARFM, GR-RL | 高频灵巧上可扩展、可复现、不绑死某一家参数化 |
| P1 | 无舰队、少人的真机闭环 | SOP 要多机；ConRFT 要人盯 | TwinRL, iRe-VLA, VLAC | 自动 reset / 恢复 / 何时问人；单机小时级且可复现 |
| P2 | 安全作为约束而不是滤镜 | SafeVLA 主要在仿真 CMDP | SafeVLA, HIL 接管 | 真机接触安全、语言安全、长尾风险，而不只是成功-碰撞折中 |
| P2 | RL 到底泛化了什么 | RL4VLA：execution ≫ vision | RL4VLA | 跨本体、跨任务、是否学到新策略而非打磨 ID 控制 |

---

## P0-1. 把 RL 经验变成可复用资产，而不是一次性改权重

**问题。** 现在几乎所有 VLA+RL 都是：SFT warm start → 某任务上 RL → 得到一个更好的 specialist checkpoint。  
ConRFT / GR-RL / BORA / SOP 都证明这条路有效，但也共同暴露同一件事：

- 更新发生在权重里，旧技能和语义对齐容易被冲掉。
- residual / LoRA / 冻结 backbone 能防遗忘，但新技能上限被 base policy 卡住。
- 真机 rollout 很贵，但学完就丢，下次换任务几乎重来。

RL4VLA 已经说明：RL 主要提高 **execution robustness**，对纯视觉 OOD 帮助有限。CRL-VLA 把稳定性-可塑性绑到 goal-conditioned advantage。真机持续学习工作则显示：异构真实演示上遗忘很实在。

**值得做的具体题目。**

1. **RL 后训练的经验去哪了？**  
   把成功 / 失败 / 干预轨迹沉淀为可检索、可回放、可编辑的资产，下次 post-training 当 prior，而不是只留下一组新权重。
2. **何时改权重、何时写记忆、何时只改 residual？**  
   需要一个可检验的判定规则：advantage 大且任务新 → 可塑；旧任务语义冲突 → 冻结或外挂。
3. **fleet / 单机反复部署下的 continual RL。**  
   不是 LIBERO 任务序列玩具，而是：同一台机器人连续一周学 4 个异构任务（抓放、接触、变形体），报告旧任务保持、新任务速度、零样本指令跟随。

**一篇好论文必须证明的事：** RL 之后 generic 能力没有塌；新经验在第三个任务上还能被调用；对比 “只存 checkpoint” 有明确复用收益。

**为什么现在做合适：** 算法派已经把 “RL 能涨点” 做熟了，缺的是 **post-training 的生命周期**，不是又一个 advantage estimator。

---

## P0-2. 让 world model 成为 RL 的可靠 simulator，而不是视频生成器

**问题。** 真机 RL 贵、仿真有 gap，所以 2026 年涌出 World-VLA-Loop / WoVR / TwinRL / VLA-RFT。真正的科学问题已经从 “画面像不像” 变成：

- 闭环节 imaginined rollout 的误差深度如何控制？
- policy 一变，world model 的 failure mode 跟着变，静态 simulator 会失配。
- 策略会不会学会 **骗模型**（假进度、假接触、假成功）？

WoVR 用 keyframe 缩短误差深度；World-VLA-Loop 用 near-success 数据和共演化；TwinRL 用显式数字孪生再把高信息状态送回真机。三条路都对，但都还没给出 **何时拒绝虚拟奖励、请求真实交互** 的原则。

**值得做的具体题目。**

1. **uncertainty-aware imagined RL：** 模型不确定或预测接触时，强制切真机 / twin，而不是一直在视频里优化。
2. **anti-exploitation 评测：** 专门构造 “模型认为成功、真机失败” 的陷阱，作为一等指标。
3. **视频模型 vs 物理孪生的分工：** 语义多样走视频；接触 / 力 / 插入走 twin 或真机。现在论文把它们当竞品，其实该是同一系统的两层 transition。

**一篇好论文必须证明的事：** 虚拟成功率上升的同时，真机成功率同步上升；关掉 “拒绝探索 / 共演化 / keyframe” 后会出现明显的模型利用。

**坑：** 只报 LIBERO + 两个真机抓香蕉，会被看成又一篇 world-model RL。

---

## P0-3. 长时序、生成式动作上的 credit assignment

**问题。** 现在主流 on-policy VLA-RL 仍把 **整条轨迹的成败** 均摊到每个 action / token / chunk。Temporal GRPO 把这叫做 trajectory-level credit aliasing：前面抓稳了、后面放失败，前面也被罚。

同时，flow / diffusion VLA 还有第二层 aliasing：信用要穿过几十步 denoising。BORA 认为这会让高维灵巧的梯度变成噪声，所以干脆 offline 学 critic、online 只训 residual。πRL 则硬做 flow 的 on-policy objective。两条路都还没成为标准。

**值得做的具体题目。**

1. **stage / subgoal credit，但不靠人工或 LLM 写死阶段。**  
   Temporal GRPO 的阶段如果可检测、可迁移，就很强；如果依赖任务脚本，就只是更好的 reward shaping。
2. **action-chunk / contact-event credit。**  
   真正该比较的不是逐步 token，而是 “这一段接触是否改变了可观测进度”。这对插拔、折叠、穿线比 LIBERO 更有意义。
3. **flow matching 的 credit 发生在哪一层：** 噪声预测、整段 chunk、还是 residual？需要消融，而不是只给一个能跑的公式。

**一篇好论文必须证明的事：** 同样稀疏终局奖励下，阶段 / 事件 credit 比轨迹均摊样本效率更高；并且阶段检测器在新任务上不崩。

---

## P1-4. 奖励：少做更密的 shaping，多做难 hack、能迁移的信号

**问题。** 稀疏成功最干净，但长时序不可学。VLA-RL 的 process RM、TGRPO 的 LLM 阶段奖励、IG-RFT 的轨迹+子任务 dense reward，都在用更密的信号换可学性，也都会引入 **reward-model bias**。SafeVLA 把安全和成功拆开，这是对的，但还在仿真。

更被低估的缺口：**VLA-RL 几乎不用接触 / 力 / 触觉作为 reward 或 constraint。** 而真机 RL 真正难的任务（插入、折叠、穿鞋带、灵巧手）失败原因经常是接触，不是语义。

**值得做的具体题目。**

1. **progress 函数当奖励，而不是手写 dense reward。** GR-RL 已经用 offline Q 当 progress 过滤演示；可以进一步把它当成在线奖励，并测量它是否比 VLM process RM 更抗 hack。
2. **多目标：成功 / 安全 / 接触质量 / 时间。** 不要把它们加权成一个标量就结束，报告 Pareto 和约束违反。
3. **人类干预信号的标准化。** APO / ConRFT / HIL 都在用接管，但 “接管 = 负例 + 纠正动作” 的 credit 还很随意。

**一篇好论文必须证明的事：** 换任务、换物体后奖励仍有效；策略没有学会刷奖励而不完成物理目标。

---

## P1-5. 面向 π / flow / 高频灵巧的原生 RL，而不是 AR 配方平移

**问题。** 仿真算法派的主体仍是 OpenVLA / OFT + PPO/GRPO。这能发论文，但和当前真正能上真机的 generalist（π0/π0.5、RDT、GR00T 的 flow / diffusion expert）正在脱节。πRL 是分水岭：token policy 的 RL **不能原样**接到 flow chunk。

灵巧、接触、毫米级任务里，人类演示往往是次优的（GR-RL 穿鞋带把这点说穿了）。这里 RL 的价值最大，也最难：动作维度高、不可逆、视觉遮挡、credit 噪声大。

**值得做的具体题目。**

1. **统一比较：** 同一 π 系 backbone 上，FPO / πRL / residual RL / RL↔SFT 交替各自的样本效率、遗忘、真机墙钟。
2. **只在需要的地方用 RL：** 低频语义策略保持 SFT，高频 expert / residual 用 RL。这比全模型 GRPO 更像可部署系统。
3. **次优演示 → 超演示专家。** 明确任务：人类做得到但不稳、不精、不快的技能。

**坑：** 只在仿真把 π0 的 LIBERO 再刷高。那是工程，不是方向。

---

## P1-6. 单机、少人、可复现的真机 RL 闭环

**问题。** 真机派已经分化成两种昂贵解：

- **人在环：** ConRFT 45–90 分钟，IG-RFT / BORA 类似，安全但不可规模化。
- **舰队：** SOP 十台机器人、云端 learner，系统正确但普通实验室进不去。

中间态才是缺口：一台或两台机器人、小时级、干预率持续下降、协议可复现。TwinRL 用 twin 找 failure region 再针对性真机 rollout，是目前最接近的思路。iRe-VLA 用 “RL 探索 + SFT 写回” 降计算和不稳定。还缺：

- 自动 reset 与失败恢复（否则永远要人）
- 何时探索、何时请求干预的策略
- 报告 cycle time、intervention rate、旧任务保持，而不只是最终成功率

**一篇好论文必须证明的事：** 同样任务上，人时和真机时显著低于 HIL baseline；关掉自动恢复后系统明显变慢或变危险。

---

## P2. 仍值得做、但更适合作为配套贡献

**安全。** SafeVLA 把问题形式化了，但真机接触安全几乎空白。如果只做仿真 cost shaping，增量不够；如果把语言约束、接触力限、恢复策略做成可评测的真机协议，价值很高。

**评测。** 现在数字不可比。做一版 “VLA+RL 诊断基准” 会很有用：execution / semantics / vision OOD、遗忘、干预率、safety cost、world-model exploitation。RL4VLA 开了头，但任务太窄、没有真机持续学习轴。

**test-time RL。** VLA-RL 提到 inference-time optimization 的苗头。它更像把计算换成功率，和 post-training 互补，但单独做成主文容易薄，适合作为系统的第三段。

---

## 怎么选：按实验室约束匹配

| 你有什么 | 更该打哪一块 | 不太该打哪一块 |
| --- | --- | --- |
| 仿真 GPU 多、真机少 | P0-2 world model 可靠性；P0-3 credit；P2 诊断基准 | 假装做了 fleets 真机 |
| 1–2 台 Franka，能 HIL | P1-6 少人闭环；P1-4 接触奖励；P0-1 连续 4 任务遗忘 | 十机器人 SOP 复刻 |
| π0.5 / flow 基座 | P1-5 原生 flow RL + residual；P0-3 denoising credit | 只在 OpenVLA-OFT 上调 GRPO |
| 触觉 / 力传感器 | P1-4 物理接地奖励/约束；接触事件 credit | 纯视觉 LIBERO 涨点 |
| 记忆 / 持续学习积累 | P0-1：RL 经验外挂记忆，而不是又一个 CL regularizer | 只在权重里做 EWC/LoRA 故事 |

一句话对照：

- 算法内卷区：LIBERO 上的 PPO vs GRPO。
- 系统正确但难复现区：舰队在线 RL。
- **论文空间最大的交叉区：RL 后训练的经验管理 × 可靠 imagined rollout × 生成式动作的 credit。**

## 若只选两个下注

1. **主线：RL 后训练的生命周期。**  
   不发明第 N 个 RL 损失，而是回答：一次真机/仿真 RL 之后，技能如何保持、如何迁移、如何避免把 generalist 打成 specialist。记忆、residual、replay、advantage 约束都可以是手段，论文主张应是生命周期，不是模块名。

2. **副线：接触敏感任务上的可靠 transition + credit。**  
   用 world model / twin 做想象，但以接触事件为信用单元，并以真机插入/折叠/灵巧证明没有骗模型。这能同时避开 “又一个抓放 GRPO” 和 “只有视频生成没有控制”。

两条线都能直接接到现有范式地图上，而且都还没有被任何一篇现有主线论文关闭。
