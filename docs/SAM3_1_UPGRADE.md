# SAM 3.1 升级审查与验证计划

审查日期：2026-10-27（上海时间）。结论：**完整同步代码，分阶段启用能力**。

## 版本与个人工作保护

- fork：`Bronya-h/sam3-industry`；审查时远端只有 `main`，头提交 `84cc43bca4347b772f17d1078a1ddb4c054655c2`。
- 上游：`facebookresearch/sam3` 的 `main`，固定到 `f1647e8d0c1d280991f4d1b7d974c2704537fe66`（2026-10-23）。
- `git rev-list --left-right --count origin/main...upstream/main` 输出 `0 60`；旧头为新头祖先；干净检出没有未提交修改。因此未发现**该远端可见分支**的个人独有提交。无法推断你其他机器上未推送的工作。
- 同步涉及 201 个文件，23410 行新增、1326 行删除（含重命名和二进制资源）。独立分支 `upgrade/sam3.1-20261027` 保留上游历史，另加本 fork 的审查和自检；不强推、不重写 `main`。
- 不建议仅摘取 Multiplex 文件：builder、decoder、memory、backbone、状态管理、后处理与性能库存在跨文件依赖；代码与新权重也必须匹配。
- 应用开关保持显式：老实验使用 `version="sam3"` 和旧权重；新视频实验使用 `version="sam3.1"` 和对应权重。图像任务继续用图像 builder。

可复查命令（从仓库根目录）：

```bash
git fetch origin
git fetch upstream
git rev-list --left-right --count 84cc43bca4347b772f17d1078a1ddb4c054655c2...f1647e8d0c1d280991f4d1b7d974c2704537fe66
git merge-base --is-ancestor 84cc43bca4347b772f17d1078a1ddb4c054655c2 f1647e8d0c1d280991f4d1b7d974c2704537fe66
git diff --stat 84cc43bca4347b772f17d1078a1ddb4c054655c2 f1647e8d0c1d280991f4d1b7d974c2704537fe66
```

## 关键差异（以代码为准）

| 范围 | 旧 fork | 新上游与影响 |
| --- | --- | --- |
| 模型结构 | 开放词汇 detector + tracker；文本/几何提示；各对象独立跟踪 | 保留原路径，增加 Multiplex tracker、mask decoder、TriHead 视觉特征路径、共享记忆编码与交互封装。不是换成通用生成式 VLM，也不是引入 RLHF |
| Object Multiplex | 对象增加带来重复的逐对象计算 | 固定容量 bucket 联合处理对象；`MultiplexState` 在对象空间与 bucket/slot 空间 mux/demux，维护 padding、删除标记和外部 object ID；`MultiplexController` 管理分组；共享记忆与多对象 decoder 减少冗余。默认每桶 16 个 slot，不代表总共只能跟踪 16 个对象 |
| 推理入口 | `build_sam3_video_predictor`，图像 `build_sam3_image_model` | 增加统一入口 `build_sam3_predictor(version="sam3.1")`（默认新版），以及 `build_sam3_multiplex_video_predictor`；显式 `version="sam3"` 路由旧模型。`max_num_objects` 与 `multiplex_count` 分别控制对象上限和桶容量，不能混为一谈 |
| 请求协议 | session、提示、视频传播 | 共享 `Sam3BasePredictor`；保留 `handle_request` / `handle_stream_request`，支持文本/点/框、mask、移除、reset/close、取消传播；stream 输出为 `{"frame_index": ..., "outputs": ...}`，mask 位于 `outputs` 内，勿照抄 builder 文档中直接读顶层 mask 的片段 |
| 状态与内存 | 原预测器实现 | 新增按需淘汰 `cached_frame_outputs` 的 streaming 开关、内存压力下清缓存和 session 管理。淘汰后不能假定仍能读取全部历史结果；需要的结果先落盘 |
| 运行优化 | 原视频推理 | 减少 CPU/GPU 同步；融合、批量后处理/视觉编码、FA3、实值 RoPE 和 `torch.compile` 路径。先 eager，再分别启用加速，预热和稳定态分开计时 |
| 权重 | `facebook/sam3` / `sam3.pt` | 新视频权重 `facebook/sam3.1` / `sam3.1_multiplex.pt`；图像与旧视频 builder 仍下载 SAM 3。新版 predictor 不应被当作无权重 CPU smoke test：未给路径会自动访问 HF |
| 训练配置 | Hydra、RF100-VL 全量图像微调、ODinW few-shot；分组学习率、DDP/SLURM | 配置 diff 主要是 BPE 路径由 `assets/` 改为 `sam3/assets/`；RF100 图像微调仍构建 `build_sam3_image_model`，默认 20 epochs、batch size 1，按 transformer/vision/language 分组 LR。没有随发布新增可直接复现的 Multiplex 视频训练 YAML；不能把旧 image YAML 换权重就宣称训练了 3.1 tracker |
| 评测 | 图像 COCO/cgF1、RF100/ODinW；视频 cgF1、pHOTA/HOTA、TETA/YTVIS 等 | 保留评测体系，更新部分实现与后处理；新增视频 notebook、速度/编译实验脚本。既有视频 YAML 不是自动使用 3.1 的保证，须核对其 builder 与 checkpoint；`bench_compile.py` 含内部生产环境说明，不是通用一键复现实验 |
| 打包 | 仅列举部分 package，词表在顶层 assets | setuptools 自动发现 `sam3*`，词表作为 `sam3/assets/*.txt.gz` 打包；NumPy 从 `==1.26` 放宽为 `>=1.26,<2` |

证据入口：[固定版本 builder](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model_builder.py)、[Multiplex 状态](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/multiplex_utils.py)、[训练配置](../sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml)、[官方发布说明](../RELEASE_SAM3p1.md)。

## 依赖与环境策略

README 最低要求在两个版本都是 Python 3.12+、PyTorch 2.7+、CUDA 12.6+；**不是这次才提高的最低门槛**。当前 README 推荐示例已由 torch 2.7.0/cu126 改成 torch 2.10.0/cu128。`pyproject.toml` 仍写 `requires-python >=3.8`，没有约束 torch/CUDA，不能只凭 pip 安装成功判断兼容。建议另建 Python 3.12 环境，按新版 README 装 GPU torch，再安装本仓库；保留旧环境，不原地覆盖。

可选 FA3/cc_torch 需要单独编译或安装检查；首次 GPU smoke 用 `compile=False, warm_up=False, use_fa3=False`，通过后再测 FA3、compile、warmup。禁用 FA3 不等于支持 CPU 推理。记录驱动、GPU、torch/torchvision、CUDA、Triton、编译器、可选扩展版本及 `pip freeze`；目前没有经过本项目实测的完整依赖锁文件。HF 权重权限需分别核实 `facebook/sam3` 与 `facebook/sam3.1`，不把 token 或权重提交到 Git。

## 已执行的离线自检

```bash
python -m unittest discover -s tests -v
```

Python 3.12.14 上 **7/7 通过**：SAM/scripts Python 源码编译、全部 notebook JSON 与新 3.1 notebook 普通 Python code cell 编译、gzip 词表与 package-data 声明、统一 builder 的默认/旧版/非法版本路由和参数转发、HF repo/文件名映射、请求移除语义、stream 分发与缓存淘汰。

测试用 AST 隔离执行**实际源文件中的路由函数**，注入 stub builder，并移除 torch decorator；不导入 SAM，不执行模型。IPython magic cell 不作 Python 编译；其他 notebook 有故意留空的路径模板，只有 JSON 校验。不能将这些结果称为 package import、完整 notebook 运行或模型数值正确性验证。

已有 `test/` 目录中的 upstream 测试与本 fork 的 `tests/` 不同；本次未执行 upstream torch 测试。当前没有 PyTorch/权重/CUDA 环境，以下均**尚未验证**：安装后的 package import、CPU tensor mux/demux、训练/反传、真实视频推理、GPU kernel/FA3、compile、显存、吞吐和数据集指标。源码编译出现 upstream `sam3/agent/agent_core.py:205` 的 tuple assert 恒真警告，本次未混入无关修复。

## 分阶段验收与实验记录

1. **安装验收**：在独立 GPU 环境安装，运行 `pip check`、`python -c "import sam3; print(sam3.__version__)"` 和 upstream `test/` 测试；另测 CPU 上随机 tensor 的 mux→demux 恒等、padding、对象新增/移除与 ID 保持。import 不等于可成功加载模型。
2. **GPU smoke**：获得两个版本权重，记录文件 SHA-256。固定一段授权短视频，显式选版本和对应 checkpoint；跑 start/add/propagate/remove/reset/close，检查 mask 尺寸、对象 ID、空检测、删除和逆向传播。先 eager + 禁用 FA3，再逐项开启优化；不将模型构建偷偷下载成功当作验收。
3. **性能 A/B**：固定 GPU、视频帧/分辨率、提示、精度、随机种子与对象 ID 初始化，分别测 1/16/32/64/128 个**实际跟踪对象**；合成圆的数量不等于 detector 找到的对象数，记录实际数。对象上限显式配置。计时前后 CUDA synchronize；分开记录加载、编译、预热、首帧、稳定态，至少 3 次重复，报告中位数/波动、峰值 allocated/reserved 显存。使用独立临时输出目录，注意 upstream speed 脚本会清空其输出目录。
4. **质量回归**：固定验证划分和提示集，复用相同 evaluator/阈值；图像 AP/cgF1，文本视频 cgF1/pHOTA 与相关 HOTA/mAP，VOS 用 J&F。不要混淆 PCS 和 VOS 协议。先保存新版 predictor 输出，再明确转换为 evaluator 要求的格式；旧视频 YAML 的模型构建不能不检查直接用。跟踪 ID、框坐标/尺寸、mask 编码、空帧都需校验。速度改善不抵消业务质量退化。
5. **微调**：从已有图像微调入口做最小过拟合实验（少量独立授权标注、固定种子、独立 train/val），记录 loss、梯度、checkpoint 恢复和 held-out 指标；确认基线后做冻结视觉/语言骨干与全量微调对照。视频 Multiplex 训练须另立任务，补齐数据 schema、时序采样、loss/matcher、可训练模块和 checkpoint 映射，先完成单 batch 反传与恢复测试，不能宣称官方 recipe 已复现。

每次实验最少记录：代码 SHA、权重 SHA-256、配置全文、环境版本、数据版本/划分、seed、硬件、命令、退出码、日志、结果文件和验证状态。先按业务容忍度约定退化阈值与延迟/显存预算，再决定是否启用新版。

官方报告（**非本次测量**）：单 H100、128 对象约 7 倍加速；YT-Temporal-1B cgF1 50.8→52.9，LVVIS mAP 36.3→34.3、BURST HOTA 44.5→43.3、YTVIS21 mAP 57.4→56.6，存在任务间取舍。完整指标及协议见发布说明，不能据此承诺你的硬件/工业视频相同收益。

## 可回滚实施

- 合并前：main 保持原样；PR 可关闭。旧代码固定在上述完整 SHA，旧环境、旧权重和配置各自保留。
- 推荐通过 **merge commit** 合并 PR，保留上游 60 个提交的谱系；合并前再次 fetch 检查 main 是否新增个人提交。如新增，先在升级分支合并并复查差异，不能覆盖。
- 验收：离线自检通过只是基础门槛；GPU 安装/推理、质量回归和性能实验通过前，保持 draft/实验状态，不切换业务默认模型。
- 合并后：在新回滚分支执行 `git revert -m 1 <本次PR的merge_commit_sha>`，提回滚 PR。`-m 1` 仅适用于上述 merge commit；若最终 squash 则用普通 `git revert <squash_sha>`。不对共享 main 执行 reset/force push。
- 立即对照旧代码可用 `git worktree add --detach ../sam3-baseline 84cc43bca4347b772f17d1078a1ddb4c054655c2`；在旧环境执行。新版代码上的 `version="sam3"` 是模型路径对照，**不等同于恢复旧版代码/依赖**。

## 如何形成可展示的实践

这次产出证明的是上游审查、依赖/接口迁移、可重复自检和回滚设计。下一步补齐小数据微调、独立验证集评测、对象规模与 compile/FA3 消融，才能形成“训练—微调—评测—推理优化”的完整项目。应分别展示已执行与计划工作；不能把同步仓库描述成自主设计模型或已完成 SAM 3.1 训练。它与你的多模态后训练方向相关，但不直接证明通用 VLM 的指令微调、偏好对齐或 RL 能力。
