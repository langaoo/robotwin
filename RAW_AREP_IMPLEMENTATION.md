# Raw AREP / k-gate 实现说明

这份笔记整理的是当前仓库里 raw AREP 这一版策略的真实实现路径，重点覆盖：

- 从数据输入到动作输出的完整前向流程
- 每一步的张量 shape
- DINOv3、DA3、FiLM 的具体作用
- raw AREP 和 k-gate 实际改了什么
- 视觉输出 1280 维到底有没有用，能不能去掉

涉及的关键实现文件：

- policy/DP2DP3/features_model/features_common/depth_guided_film_online/extractors_2model.py
- policy/DP2DP3/features_model/features_common/depth_guided_film_online/encoder_film_2model.py
- policy/DP2DP3/features_model/features_common/depth_guided_film_drifting/policy_drifting.py
- policy/DP2DP3/features_model/features_common/depth_guided_film_drifting/policy_drifting_v18_raw_arep.py
- policy/DP2DP3/features_model/tools/depth_guided_film_drifting/train_film_drifting.py
- policy/DP2DP3/deploy_film_drifting_policy.py
- policy/DP2DP3/features_model/configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_lift_pot_seed42.yaml

## 1. 总体结构

这版 raw AREP 策略可以拆成三段：

1. 视觉 token 提取：DINOv3 提语义 token，DA3 提几何 token
2. 融合编码：用 DA3 的全局几何向量通过 FiLM 调制 DINOv3 token
3. 动作生成：把融合后的视觉特征和 proprio 拼起来，送进一个条件动作头，输出 8 步、14 维动作轨迹

其中 raw AREP 不是视觉分支的改动，而是动作空间里 drifting target 的定义方式变了。

## 2. 配置里的关键超参

以当前 lift_pot 的 raw AREP 配置为例：

- horizon = 8
- n_obs_steps = 3
- n_action_steps = 6
- proj_dim = 256
- out_dim = 1280
- temp_scales = [0.5, 1.0, 2.0]
- K = 32
- drift_scale = 50.0
- arep_min_alpha = 0.50
- arep_alpha_tau = 1.0

含义是：

- 每次看 3 帧观测
- 每次预测 8 步动作
- 部署时通常只执行前 6 步
- FiLM 内部统一工作在 256 维
- 每个观测时刻最终压成一个 1280 维视觉向量

动作维度是 14：

- 左臂 6
- 左夹爪 1
- 右臂 6
- 右夹爪 1

## 3. 训练数据如何组织

训练入口会从 zarr 中读取：

- data/head_camera
- data/state
- data/action
- meta/episode_ends

其中：

- RGB 序列可以理解为 [N, 3, H, W]
- state 是 [N, 14]
- action 是 [N, 14]

预提 token 之后，CachedTokenDataset 会切训练窗口。

对单个样本，返回：

- dino_tokens: [To, Kd, 768]
- da3_tokens: [To, Ka, 2048]
- agent_pos: [To, 14]
- action: [H, 14]

这里：

- To = n_obs_steps = 3
- H = horizon = 8

如果 batch size 是 B，则训练时进入策略前的 shape 是：

- DINO tokens: [B, 3, Kd, 768]
- DA3 tokens: [B, 3, Ka, 2048]
- agent_pos: [B, 3, 14]
- actions: [B, 8, 14]

注意动作标签不是和观测对齐的同一时间片，而是从观测窗口后面开始取 8 步未来动作。

## 4. 视觉 token 提取

### 4.1 DINOv3 分支

DINOv3 负责语义。

单张图像经过 resize 和 normalize 后，抽 patch token。

默认配置下通常是：

- 输入：RGB image
- 输出：[B, Kd, 768]

其中 Kd 一般接近 14 x 14 = 196 个 patch token。

### 4.2 DA3 分支

DA3 负责几何。

它不是直接输出单个 pooled feature，而是取 backbone 某层 feature，再 flatten 成 token：

- 输出：[B, Ka, 2048]

Ka 不一定天然等于 196，但后面会统一 subsample 到最多 196 个 token。

### 4.3 两个分支的语义分工

- DINOv3：偏语义，告诉模型看到了什么
- DA3：偏几何，告诉模型结构和空间关系

所以这个设计不是两个视觉特征简单拼接，而是“几何条件化语义”。

## 5. FiLM 融合编码器怎么工作

融合编码器是 DA3Film2ModelEncoder。

输入：

- x[0] = DINO tokens: [B, To, K_sem, 768]
- x[1] = DA3 tokens: [B, To, K_geo, 2048]

### 5.1 展平时序

先把 B 和 To 合并：

- sem_flat: [B x To, K_sem, 768]
- geo_flat: [B x To, K_geo, 2048]

记 N = B x To。

### 5.2 Stage 1：统一投影到 256 维

DINO tokens 走：

- Linear 768 -> 256
- LayerNorm

得到：

- q_tokens: [N, K_sem, 256]

DA3 tokens 走：

- Linear 2048 -> 256
- LayerNorm

得到：

- geo_proj: [N, K_geo, 256]

然后两个分支都会等间隔 subsample 到最多 196 个 token。

### 5.3 Stage 2：FiLM 调制

DA3 token 先在 token 维上做 mean pool：

- geo_vec = mean(geo_proj, dim=1)
- shape: [N, 256]

这个几何向量进入 FiLM MLP：

- 256 -> hidden 256 -> 512

再切成：

- scale: [N, 256]
- shift: [N, 256]

然后对每个 DINO token 做逐通道调制：

fused = semantic_token x (1 + scale) + shift

因此 shape 不变，仍然是：

- q_tokens: [N, K, 256]

这一步的本质是：

- DINO 给语义 token
- DA3 给条件向量
- 几何信息不是和语义简单拼接，而是通过条件仿射变换作用在语义 token 上

### 5.4 Stage 3：token 池化得到每时刻视觉 embedding

FiLM 后的 token 再做 mean pool：

- pooled: [N, 256]

再经过 output_proj：

- Linear 256 -> 2560
- GELU
- Dropout
- Linear 2560 -> 1280
- LayerNorm

最终得到：

- z: [N, 1280]

再 reshape 回时序：

- [B, To, 1280]

这就是视觉分支给动作头的最终输入。

## 6. 1280 维之后怎么进入动作头

每个观测时刻都有：

- 视觉特征 1280 维
- proprio 14 维

拼起来得到每时刻：

- 1294 维

3 个观测时刻平铺后：

- 3 x 1294 = 3882 维

然后进入 obs_encoder：

- Linear 3882 -> 512
- ReLU
- Linear 512 -> 256
- ReLU

最终得到全局条件向量：

- obs_cond: [B, 256]

这个 256 维条件才是真正给 ConditionalUnet1D 的条件输入。

所以要分清两层：

- 视觉输出接口维度是 1280
- 动作头真正消费的全局条件维度是 256

## 7. 动作头是什么

动作头不是 Transformer，而是 ConditionalUnet1D。

输入给动作头的是：

- 一个初始噪声轨迹 [B, 8, 14]
- 固定 timestep = 0，shape [B]
- 全局条件 obs_cond: [B, 256]

输出：

- 预测动作轨迹 [B, 8, 14]

推理时它只做一次 forward，不做多步扩散采样，所以是 1 NFE。

## 8. drifting loss 在做什么

这版训练不是 DDPM 式噪声预测，而是 drifting target 回归。

动作轨迹展平后：

- D = 8 x 14 = 112

对每个观测，采 K 个粒子。当前配置 K = 32。

训练时：

- noise_k: [B x 32, 8, 14]
- gen_k: [B x 32, 8, 14]
- gen_flat_k: [B, 32, 112]
- pos_flat: [B, 112]

然后构造漂移场 V：

- V: [B, 32, 112]

目标定义为：

- target = gen + drift_scale x V

最后做 MSE(gen, stopgrad(target))。

## 9. raw AREP 和普通 drifting 的区别

raw AREP 改的不是视觉编码器，也不是动作头结构，而是漂移场 V 的计算方式。

它继承多温度 drifting 版本，但使用 proximity-aware asymmetric repulsion。

仓库里没有完整写出 AREP 缩写，不过从实现和注释看，它对应的是 asymmetric repulsion，一种靠近 expert 时削弱粒子间排斥的机制。

### 9.1 为什么要做 AREP

普通 drifting 有一个问题：

- 生成粒子彼此之间会排斥
- 当粒子已经靠近 expert action 时，这个排斥仍然可能太强
- 结果是它们不容易继续塌缩到 expert 附近

AREP 的目的就是：

- 远离 expert 时保留排斥，维持多样性和场的稳定性
- 靠近 expert 时减弱 gen-gen repulsion，让粒子更容易贴近 expert

## 10. k-gate 是什么

这里的 k-gate 不是单独的一层网络，也不是视觉门控。

它指的是 kernel-gated repulsion，也就是在 kernel 层面对 gen-gen block 做门控。

代码逻辑是：

1. 对每个观测，把 expert action 拼到 target 集合末尾
2. 计算生成粒子到所有 target 的距离矩阵
3. 用粒子到 expert 的距离定义每个粒子的 alpha
4. 只对 gen-gen kernel block 加门，不改 gen-pos kernel

alpha 的定义是：

- 如果粒子离 expert 很近，alpha 接近 min_alpha
- 如果粒子离 expert 很远，alpha 接近 1

当前配置：

- min_alpha = 0.5
- alpha_tau = 1.0

gen-gen block 的门控是对称的：

- k_gg(i, j) <- sqrt(alpha_i x alpha_j) x k_gg(i, j)

而 gen-pos 那一列不变。

这意味着：

- 靠近 expert 的粒子之间，排斥会被削弱
- 但它们对 expert 的吸引不被削弱
- 同时尽量保留原 drifting field 的固定点性质

这就是 checkpoint 名字里 kgate 的含义。

## 11. raw AREP 的多温度聚合

当前配置用了三个温度：

- 0.5
- 1.0
- 2.0

raw 的意思是：

- 各温度分支分别算漂移场
- 不做每个分支的单独归一化
- 直接平均

也就是：

- V = mean(V_0.5, V_1.0, V_2.0)

再乘 drift_scale = 50 得到最终位移。

## 12. 推理时从输入到输出的 shape 流程

部署时每次环境给的是单帧观测：

- head_cam: [3, H, W]
- agent_pos: [14]

wrapper 内部维护一个长度为 3 的 obs_buffer。如果还不到 3 帧，就复制第一帧补满。

### 12.1 在线提 token

最近 3 帧图像送进 TwoModelExtractors，得到：

- DINO: [3, Kd, 768]
- DA3: [3, Ka, 2048]

再补 batch 维：

- DINO: [1, 3, Kd, 768]
- DA3: [1, 3, Ka, 2048]
- agent_pos: [1, 3, 14]

### 12.2 融合编码

经过 FiLM encoder 后：

- fused: [1, 3, 1280]

拼 proprio：

- [1, 3, 1294]

flatten：

- [1, 3882]

obs_encoder 后：

- obs_cond: [1, 256]

### 12.3 动作生成

采样 noise：

- [1, 8, 14]

固定 timestep=0：

- [1]

ConditionalUnet1D 输出：

- [1, 8, 14]

再做反归一化：

- [1, 8, 14]

squeeze 后：

- [8, 14]

部署代码会再 clip 到 [-3, 3]，然后通常只执行前 6 步，所以实际返回给环境的是：

- [6, 14]

## 13. 1280 到底有没有用

这是一个必须分层回答的问题。代码能证明一部分，不能证明另一部分。

### 13.1 代码能证明的事实

可以确定的事实有三条：

1. FiLM 真正工作的核心维度是 256，不是 1280
2. 1280 出现在 token 池化之后，是视觉分支输出给策略头的接口维度
3. 后续 obs_encoder 的输入维度直接依赖这个 1280

换句话说：

- 256 是融合计算的内部工作维度
- 1280 是融合结果导出给动作策略的表示维度

因此，1280 不是 FiLM 必需的，也不是 DINO/DA3 backbone 必需的。它是一个 post-fusion representation size。

### 13.2 1280 在数学上做了什么

如果只看 encoder 内部，FiLM 结束后已经有：

- pooled: [N, 256]

这时其实已经可以直接给策略头。

但当前实现没有这么做，而是又经过一个 MLP：

- 256 -> 2560 -> 1280

这一步相当于做了一个非线性重编码，把 token 平均池化后的 256 维摘要重新映射到更高维空间，再交给下游策略使用。

它可能带来的好处是：

- 提高下游条件表示的容量
- 给 obs_encoder 更多线性可分的特征组合
- 与仓库中其他实验保持统一接口

但注意，这里我说的是“可能”。代码里没有 ablation，所以不能严肃地宣称 1280 一定更优。

### 13.3 工程上 1280 的真实意义

从工程角度看，1280 有三个实际作用：

1. 统一接口
   注释里已经写了 out_dim = 1280, 与其他实验对齐。这意味着很多实验和 checkpoint 默认都以 1280 作为视觉输出接口。

2. 扩大下游条件容量
   obs_encoder 的第一层输入是 3 x (1280 + 14) = 3882。如果 out_dim 更小，这层看到的视觉信息维度会直接下降。

3. 让视觉分支和动作头解耦
   视觉分支内部始终在 256 维里做 FiLM 融合，但对外暴露 1280 维接口。这样可以在不动 FiLM 主体的前提下，给不同下游头保留更大的适配空间。

### 13.4 它是不是“必须的”

不是必须的，但对当前 checkpoint 是必须兼容的。

要分两种情况：

#### 情况 A：你想直接拿现有 checkpoint 推理

那 1280 不能去掉。

原因很简单：

- fusion_encoder.output_proj 的权重 shape 会变
- obs_encoder 第一层的输入 shape 会变
- checkpoint load 会直接不匹配

所以对已有模型，1280 不是可选项，而是接口契约的一部分。

#### 情况 B：你愿意重新训练

那 1280 可以改，甚至可以去掉。

但这里的“去掉”有两种不同力度：

1. 温和改法：把 out_dim 从 1280 改小，比如 512 或 256
2. 激进改法：彻底移除 output_proj，让 pooled 256 直接输出给策略头

这两种都可行，但都必须重新训练，旧 checkpoint 不能直接复用。

### 13.5 如果把 1280 改成 256，会省多少

按当前网络结构估算：

- output_proj 在 1280 设定下约 3,938,560 个参数
- obs_encoder 第一层在 1280 设定下约 1,988,096 个参数

如果改成 256：

- output_proj 约 263,424 个参数
- obs_encoder 第一层约 415,232 个参数

两部分合计可减少：

- 5,248,000 个参数

这是实打实的节省，不是口头上的“可能会变轻”。

### 13.6 去掉 1280 后速度会快多少

会快，但通常不是决定性提升。

原因是这套系统的重头主要在：

- DINOv3 backbone
- DA3 backbone
- ConditionalUnet1D

而 1280 相关的部分主要是：

- 一个后处理 MLP
- obs_encoder 的第一层输入更宽

所以：

- 参数量会明显下降
- 显存会下降一些
- 前向会更快一些
- 但如果你的瓶颈主要在视觉 backbone，提速不会和参数节省成正比

### 13.7 去掉 1280 的风险是什么

真正的风险不是“会不会跑不起来”，而是表达能力是否掉得太多。

因为一旦去掉 1280，相当于你让策略头直接吃 256 维视觉摘要。这样做的代价可能是：

- 视觉摘要更紧
- obs_encoder 可利用的特征组合更少
- 动作头更依赖 256 维 pooled feature 的质量

尤其这里用了 simple mean pool，token 时空信息本来已经压缩得很狠。1280 这层某种程度上是在池化后再补一次表达容量。

所以严格的结论应该是：

- 1280 不是理论必需
- 1280 是当前实现里一个有现实工程意义的容量扩展层
- 能不能删，取决于你更在意兼容性、速度还是最终效果

### 13.8 我的判断

如果你的目标是：

- 先把现有 raw AREP 跑稳
- 保持和已有 checkpoint、配置、评测结果兼容

那不要动 1280。

如果你的目标是：

- 做压缩版实验
- 测试 256 是否足够
- 降参数、降显存

那 1280 完全可以作为 ablation 项来改，而且这是合理的改动，不是乱删。

但更合理的顺序不是“直接删掉”，而是：

1. 先把 out_dim 从 1280 改成 512
2. 再试 256
3. 最后再试彻底去掉 output_proj

这样你能分清楚：

- 是高维接口重要
- 还是 output_proj 这个非线性映射本身重要

## 14. 结论

最核心的几点：

1. raw AREP 没改视觉编码器，视觉仍然是 DINOv3 语义 token + DA3 几何 FiLM 调制
2. kgate 不是新网络，而是动作空间 drifting field 里的 kernel-gated gen-gen repulsion
3. 1280 不是 FiLM 必需维度，它是视觉分支导出给动作头的接口维度
4. 对已有 checkpoint 来说，1280 不能随便删，因为它是 shape 契约的一部分
5. 如果愿意重训，1280 可以缩小甚至移除；把它从 1280 改到 256，大约能减少 524.8 万个参数

## 15. 如果后续要继续做实验

建议按下面顺序做，而不是一步到位直接删：

1. out_dim: 1280 -> 512
2. out_dim: 512 -> 256
3. 删除 output_proj，直接输出 pooled 256

并分别比较：

- 训练稳定性
- 成功率
- 推理耗时
- GPU 显存
- 对不同任务的泛化差异

只有这样，才能判断 1280 在你这套任务上到底是“真有贡献”，还是“只是历史兼容值”。