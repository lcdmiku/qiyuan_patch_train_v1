# 无人机航拍图搜图定位 — 提交方案

> 128×128 查询图块 → **任务A** 检索来源大图（R@1/R@5）、**任务B** 在大图中定位（Acc@0.5/Acc@0.75）
> `总分 = 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75`

---

## 一、方案核心

**把两个任务统一成一个原语：归一化互相关（NCC）模板匹配。**

依据是一条实测事实：**查询图块是来源大图的精确像素裁剪**，逐像素差为 0（train/val 各抽样验证）。
因此图块就是原分辨率下的模板，NCC 是**精确且最优**的匹配器 —— 不需要学习式配准、不需要特征描述子。

真正的工程难点不在精度而在**成本**：全库全分辨率搜索远超 60 分钟预算。方案用**粗到细级联**解决：

```
图块 (128×128)
    │
    ├─ 级联筛选 ──► ds=8 扫全库 154 张   (0.7 s)   → 保留 24
    │               ds=4 扫 24 张        (1.6 s)   → 保留 8
    │
    └─ 全分辨率精修 ──► 在种子位置周围 256×256 窗口搜索 (±64px 余量)
                        命中即返回精确像素                    (0.03 s)
```

**精度来源**：粗筛只需把真值定位到 ±4 px（级联在 ds=4 的命中率 96.7%），
全分辨率窗口精修负责从该种子恢复**精确像素**。实测 297/297 精确命中真值。

---

## 二、验证结果

在赛题自带 **val 划分全量（296 个图块 / 77 张大图）** 上，用赛题同款指标评测：

| 指标 | 数值 |
|---|---|
| R@1 | **1.0000** |
| R@5 | **1.0000** |
| Acc@0.5 | **1.0000** |
| Acc@0.75 | **1.0000** |
| **总分** | **1.0000** |
| 平均 IoU | **1.0000（100% 像素级完全重合）** |
| 定位路径 | `window` 296 / 296 —— 全部一次命中，兜底分支从未触发 |
| 错图计数 | 0 / 296 |
| 耗时 | 解码 7.4 s + 求解 1798 s = **6.08 s/图块**（RTX 4060 8GB 笔记本 GPU） |

独立交叉验证：`run.py` 端到端跑合成输入（20 查询，5 检索 + 15 定位），用独立打分脚本
`tools/score_pred.py` 核对 —— 同样 **1.0000**，15/15 精确定位。

### 时间预算

单查询 6.08 s 是 **8GB 笔记本 GPU** 的成绩。评测环境是 A100-80GB，显存宽裕使 `unfold`
矩阵乘无需分块，预期快 3–5 倍。即使按当前最保守速度：

```
340 查询 × 6.08 s + 154 图解码 ≈ 35 分钟   <   60 分钟预算
```

`run.py` 另有 50 分钟软预算（`TIME_BUDGET` 可调）与"定位优先"调度兜底。

---

## 三、目录结构

```
run.py                      评测入口（python /workspace/run.py）

patchmatch/                 主包，严格分层单向依赖
├── core/                   layer 1：几何 + NCC 数值核 + 图像金字塔缓存
│   ├── geometry.py         bbox 几何、IoU
│   ├── ncc.py              ★ NCC 核心（含分块 unfold+GEMM、块平均降采样）
│   └── imagebank.py        图像解码 + 金字塔 GPU 缓存
├── models/                 layer 2：学习式描述子（可选第二意见）
│   ├── descriptor.py       PatchDescriptorNet：ResNet 式主干 + GeM 池化
│   ├── losses.py           Circle loss + batch-hard triplet
│   ├── dataset.py          数据集 + 类平衡采样
│   └── train_descriptor.py 训练脚本
├── retrieval/              layer 3：任务A
│   ├── cascade.py          ★ 多级粗到细筛选
│   └── scorers.py          NCC / 描述子打分器 + 排名融合
├── grounding/              layer 4：任务B（占 0.7 分）
│   └── locate.py           ★ 三段式定位
├── infer/                  layer 5：编排与 I/O 契约
│   ├── input_reader.py     解析 /input_path（容忍字段别名）
│   ├── output_writer.py    写 pred.json
│   ├── solver.py           主管线（先定位后检索）
│   └── embedding_index.py  描述子图像嵌入
└── eval/                   layer 6：本地评测
    └── harness.py          完整指标评测

tests/                      单元测试（NCC 对照暴力参考、相位失配回归）
tools/                      性能剖析与数据准备脚本
Dockerfile                  镜像构建
ARCHITECTURE.md             详细架构文档
```

---

## 四、运行

### 本地验证

```bash
# 1. 构造合成输入（复用训练包，模拟 /input_path 布局）
python tools/make_synthetic_input.py --split val --out .local_input --retrieval 5 --grounding 15

# 2. 跑完整管线
python run.py --input .local_input --output .local_output

# 3. 对照真值打分
python tools/score_pred.py --input .local_input --pred .local_output/pred.json
```

### 完整评测

```bash
python -m patchmatch.eval.harness --data qiyuan_patch_train_v1 --split val
```

### 构建镜像

```bash
docker build -t qiyuan-patchmatch:v1 .
docker save qiyuan-patchmatch:v1 | gzip > submission.tar.gz
```

---

## 五、评测环境行为

| 项 | 说明 |
|---|---|
| 入口 | `python /workspace/run.py`（镜像 `ENTRYPOINT` 已设，可直接跑） |
| 输入 | `/input_path/`（可用 `INPUT_PATH` 环境变量或 `--input` 覆盖） |
| 输出 | `/output_path/pred.json`（可用 `OUTPUT_PATH` 或 `--output` 覆盖） |
| 断网 | 依赖全在构建期装入；**方案无需任何权重下载**，NCC 管线免训练 |
| 超时保护 | 默认 50 分钟软预算（`TIME_BUDGET` 可调）；**定位查询全部先做完**再处理检索，确保 0.7 分的高权重任务优先完成 |
| 容错 | 每个查询独立 try；失败则输出合法兜底答案（检索给库序、定位给居中框），**任何情况下都会写出 pred.json** |
| GPU | 自动检测；无 GPU 时回落 CPU（可跑通，但慢） |

---

## 六、关键工程决策（含踩过的坑）

完整清单见 `ARCHITECTURE.md`。最值得记录的三条：

1. **降采样必须用块平均，不能用跨步采样。** 裁剪偏移 `x=1978`（`mod 4 = 2`）时，
   图块的 `[::4]` 采样栅格与大图的 `[::4]` 栅格**错开 2 像素**，同一内容不再对应，
   真值处 NCC 甚至为**负**（−0.03，即负相关）。改用块平均后真值 NCC 转正，R@1 由 0.975 → **1.000**。
   回归测试见 `tests/test_downsample.py`。

2. **GPU 上 float64 卷积比 float32 慢 90 倍**（cuDNN 无快速 fp64 实现：1141 ms vs 33.5 ms）。
   但局部方差的 `E[X²]−E[X]²` 在 float32 下会灾难性抵消（8bit 像素均值≈128 时两项都 ~4e4，
   差值仅数百，float32 只剩 1–2 位有效数字）。折中：**相关和用 float32，窗口均值/方差仍用 float64**。

3. **`conv2d` 的 batch 维不是"每样本一个卷积核"。** PyTorch 会静默用 kernel 0 卷积所有
   batch 元素，K=1 时完全看不出来。因此多模板搜索走 `unfold` + 分块 GEMM（同时快 5–16 倍，
   且与暴力参考误差 5e-15，比逐模板卷积路径的 1e-8 还准）。

还有一类更容易被忽略的坑：**默认值重复定义**。`harness.py` 的 argparse 把 `--levels` 默认值
写死为 `"8,4,2"`，而 `CascadeConfig` 里是调优过的 `(16,8,4)`；由于 `evaluate()` 总会传入显式值，
配置里的调优默认值**永远不生效**，导致实际跑的是慢配置（102 s/查询 vs 1.6 s）。
已改为从配置对象派生默认值。

---

## 七、关于学习式分支

`patchmatch/models/` 实现了完整的图块描述子网络（ResNet 式主干 + GeM 池化 + Circle loss
度量学习），作为检索的**第二意见**接入（按排名融合，量纲无关）。

**但它对当前数据的边际价值为零**，原因很直接：图块与来源图是精确裁剪关系，NCC 在这个任务上
是精确最优的，学习式描述子不可能超过一个"精确匹配器"。它的真实价值是**保险**——若评测图块
并非精确裁剪（重压缩、重采样、色彩偏移），NCC 会退化得比学习式嵌入更快。

因此当前提交**不打包任何权重**（镜像仅数百 MB），保持零训练依赖。若需要启用，
用 `train_descriptor.py` 训练后在 `solver.attach_descriptor()` 注入即可。
