# OpenSearch kNN 边写边查去重 · 调优最佳实践

> 面向本项目的去重场景：**边写边查、索引持续增长**——每来一批向量，先对现有索引做
> kNN 检索判重，没命中才写入。索引在被查询的同时不断变大，`search` 阶段会随规模变慢。
> 本文汇总经压测验证的调优要点、实测数据趋势，以及大规模（亿级 / 十亿级）的架构建议。
>
> 本项目 `src/config.py` 的默认值已按下述最佳实践设置。

---

## 0. 一句话结论

- **绝对吞吐低** 多半是**集群没喂饱**（检索并发太低 / 过度分片 / 客户端编码瓶颈），不是 HNSW 算力不够——先定位再优化。优先安装 `orjson`（客户端序列化提速 5-10×）。
- **随规模变慢** 主要是 HNSW 的 **log(N) 本征增长** + segment 叠加，属**温和退化**（不会断崖），可用刷新/合并/分片策略压平。
- **7B 规模** 下"单索引 + 边写边查 + 单 worker 顺序"这个形状会顶不住 deadline，应认真评估**两阶段离线去重**（见第 6 节）。

---

## 1. 关闭周期性自动刷新，只靠每批显式 refresh ⭐

**问题**：`refresh_interval` 默认 `1s` 时，一批几十秒的 `search`+`write` 期间，后台会触发
几十次自动刷新，**每次刷新都切出一个新 segment**。Faiss/Lucene 的 HNSW 图**按 segment 独立存在**，
一次 kNN 查询要遍历**每一个 segment 的小图**——segment 越积越多，`search` 越来越慢（segment 膨胀）。

**最佳实践**：`refresh_interval = -1`（禁用自动刷新）。本项目在每批写入后调用**同步的**
`index.refresh()`，可见性完全由它保证——下一批一定看得到上一批。中途的自动刷新对正确性无贡献，
纯属制造碎 segment。

```bash
export OS_REFRESH_INTERVAL=-1   # 默认值；设 "1s" 可做 A/B 对比
```

> 显式 `index.refresh()` 是**同步**调用（返回即刷新完成、文档可见），每批末尾**无需再 sleep**。
> `REFRESH_WAIT_S` 默认 `0`。

---

## 2. 去重只需 top-1，用小的 ef_search ⭐

**问题**：`knn.algo_param.ef_search` 偏大（如 256）对每次查询是纯 CPU 浪费——去重只需判断
"最近 1 个邻居是否 ≥ 0.95"，不需要高召回 Top-K 排序。

**最佳实践**：`ef_search = 32`（默认）。若准确率统计显示近重复漏检增多，再上调到 64。

```bash
export OS_EF_SEARCH=32   # 默认值；漏检偏多时试 64
```

> ⚠️ `ef_search` 是**索引级设置**，仅在**创建索引时**写入。改后必须**重建索引**才生效。

---

## 3. 分片数匹配"每片数据量"，避免过度分片 ⭐

**问题**：kNN 查询会 **fan-out 到所有分片**，每片各做一次 HNSW 搜索再合并。对固定数据集，
**分片越多、单查询要做的 HNSW 小图搜索越多**。例如 10M 向量用 16 分片，每片仅 ~62 万，
每个查询白白多做约 8 倍的小图搜索，还多了协调开销。

**最佳实践**：按**每片约 5–7M 向量**（生产每片约 75 GB）定分片数：

| 数据规模 | 建议主分片数 |
|---|---|
| 1000 万 | 2 |
| 1 亿 | ~16 |
| 70 亿 | ~1,000–1,200 |

```bash
export OS_SHARDS=2   # 10M 测试建议值
```

---

## 4. 激进的段合并策略，减少每查询要遍历的 HNSW 图数 ⭐

**问题**：即使关了自动刷新，每批显式 refresh 仍会产生 segment；默认合并策略较保守，段数会积累。
段越多，每次 kNN 查询要遍历的独立 HNSW 图越多。

**最佳实践**：建索引时用更激进的合并策略，让段尽快合并成更少、更大的段：

```json
"merge.policy.max_merge_at_once": 20,
"merge.policy.segments_per_tier": 5,
"merge.policy.floor_segment": "50mb"
```

```bash
export OS_MERGE_MAX_AT_ONCE=20
export OS_MERGE_SEGMENTS_PER_TIER=5
export OS_MERGE_FLOOR_SEGMENT=50mb
```

> ⚠️ 同为**索引级设置**，改后需重建索引。
>
> **关于 batch 大小与 segment**：segment 产生频率由 **refresh 频率**（每批一次）决定，**不由 batch 大小决定**。
> 小 batch（如 20K）单次刷出的段更小，反而**更容易被 `floor_segment`/`segments_per_tier` 快速合并掉**；
> 大 batch（如 100K）只是让本地去重变慢（100K×100K 矩阵 ~20s vs 20K ~1s），对 segment 数几乎无帮助。
> **推荐 batch=20K**。

---

## 5. 定位瓶颈：先测量，别靠猜

### 5.1 用分阶段计时看哪一段在变慢

`--report-every-batch` 逐批打印 `local / search / write / refresh` 各阶段耗时：

```
batch  1 | n=20,000 | local 1.0s  search  5.7s  write 5.9s  refresh 1.1s | ...
batch 140| n=20,000 | local 1.2s  search 16.2s  write 5.9s  refresh 1.3s | ...
```

- `search` 随 `idx_total` 上涨 → 本文 1/2/3/4 全部适用。
- `write` 随规模上涨 → 关注 segment merge、分片写入均衡、bulk 并行度。
- 各阶段都稳定但**绝对值高** → 看 5.2 是不是集群没喂饱。

### 5.2 关键判定：瓶颈在客户端还是集群？

`_msearch` 返回里每条有 `took`（服务端毫秒）。**把一批所有 `took` 之和 与 客户端测到的
`search` 秒数 对比**：

- server `took` 之和 **≪** 客户端 `search` 时间 → 瓶颈在**客户端**（JSON 编码受 Python GIL 串行、检索并发太低）。**加集群节点没用**，应先加 `MSEARCH_WORKERS`、换 `orjson`、或多进程。
- 两者接近 → 瓶颈真在**集群**（分片过多 / 索引太大 / 节点不足）。

> 经验：若"每核每秒 kNN 查询数"只有个位数，几乎一定是没喂饱——ef_search=32 的单核 HNSW 应能到上千/秒。

### 5.3 检索并发要打满集群

`MSEARCH_WORKERS` = 并发检索连接数。**它太低（如 10）会让几百 vCPU 的集群大部分空闲**。
起步按 **≈ 集群总 vCPU 数量级**去调（如 512 vCPU 集群设 64–128），观察节点 CPU 是否被打满。

---

## 6. 客户端序列化优化：用 orjson 替代标准 json ⭐

**问题**：`_msearch` 请求体需要将每条 768 维 FP32 向量序列化为 JSON。Python 标准库 `json.dumps`
是纯 Python 实现，受 GIL 限制，序列化大量浮点数组时成为**客户端瓶颈**——即使开 64 个线程，
实际只有 1 个 CPU 核在做 JSON 编码。

**实测对比**（10M 测试，batch=40K, MSEARCH_CHUNK=500, MSEARCH_WORKERS=64）：

| 序列化库 | batch 1 search | batch 4 search | vps |
|---------|---------------|---------------|-----|
| 标准 json | 10.64s | 15.19s | ~1,330 |
| **orjson** | **1.73s** | **2.82s** | **~2,150** |
| **提升** | **6.2×** | **5.4×** | **+62%** |

**最佳实践**：安装 `orjson`（C 实现的高性能 JSON 库），代码自动使用：

```bash
pip install orjson>=3.9.0
```

```python
# os_client.py 中的实现（已集成）
try:
    import orjson
    _fast_dumps = lambda obj: orjson.dumps(obj).decode("utf-8")
except ImportError:
    _fast_dumps = json.dumps  # fallback 到标准库
```

**原理**：
- `orjson` 用 Rust/C 编写，序列化速度是标准 `json` 的 5-10 倍
- 对 numpy float 数组序列化尤其高效
- 不受 Python GIL 限制，线程并发时不互相阻塞
- 向后兼容：如未安装 orjson 则自动 fallback 到标准 json

**配合 MSEARCH_CHUNK 和 MSEARCH_WORKERS 使用**：

解决了客户端编码瓶颈后，可以进一步缩小 chunk、增大并发，让集群 CPU 被充分利用：

```bash
export MSEARCH_CHUNK=500       # 从默认 1000 减半，更多并行粒度
export MSEARCH_WORKERS=64      # 按集群 vCPU 规模上调
```

| 配置 | search 阶段耗时 | 说明 |
|------|---------------|------|
| json + CHUNK=1000 + W=32 | ~15s/批 | 客户端编码瓶颈 |
| orjson + CHUNK=1000 + W=32 | ~5s/批 | 编码快了，但 chunk 数少 |
| **orjson + CHUNK=500 + W=64** | **~3s/批** | 编码快 + 并发打满 |

> 判断方法：若集群 data node CPU 仅 20-30%，说明客户端没喂饱。用 orjson + 更高并发后
> 集群 CPU 应上升到 50-70%，吞吐同步提升。

---

## 7. 大规模（亿级 / 十亿级）的架构建议

### 6.1 退化是温和的，但吞吐天花板是真的

实测（10M、16 节点、ef_search=32）：`search` 随索引 7万→267万 增长约 **3×**，而数据量涨了约 38×——
**接近 log(N)，不是线性**，说明退化温和、不会断崖。按此外推到 7B，每批 `search` 大致从几十秒
涨到百余秒，仍可运行，但**总耗时会顶到 deadline**。

### 6.2 7B 的正解：把"插入"和"查询"解耦（两阶段离线去重）

当前"查到没重复才插"的耦合，强制了串行 + 每批 refresh + 查一个不断变大的索引。规模一大就顶不住。
推荐改为两阶段：

1. **阶段一 · 全量灌入不去重**：用 **GPU 加速摄取**把全部 raw 向量写进 OpenSearch，纯写入拉满吞吐。
2. **阶段二 · 只读并行去重**：索引写完后 **force-merge 一次**（段数最少、检索最快），再对每条查 top-1 近邻。
   这是**只读、静态索引、无 refresh、无竞态**，可用**几百个并行 worker** 打满集群，想加节点就加。
3. **阶段三 · 删除标记的重复**（同一近邻簇保留最小 id，语义正确）。

**收益**：查询阶段索引不再增长 → `search` 恒定；只读查询可无限并行、无竞态、无需单 worker；
写入与查询各自打满，不再互相拖累。**代价**：一次全量存储（raw 而非去重后）+ 一个 tie-break 规则。

### 6.3 内存与量化

- HNSW 内存 ≈ `1.1 × (4×dim + 8×m) × N`；768 维、m=16、FP32 下 **100M ≈ 352 GB**，7B 需 TB 级。
- 大规模务必配合**标量量化 / FP16**（4x/2x 压缩）与 `method: memory_optimized`（mmap page cache）。

---

## 8. 调优默认值一览（本项目已按最佳实践设置）

| 参数 | 默认 | 说明 |
|---|---|---|
| `OS_REFRESH_INTERVAL` | `-1` | 禁用自动刷新，只靠每批显式 refresh |
| `REFRESH_WAIT_S` | `0` | 显式 refresh 已同步，无需额外 sleep |
| `OS_EF_SEARCH` | `32` | 去重 top-1 够用，省 CPU |
| `OS_SHARDS` | `8` | **按数据量调整**：10M→2，1 亿→~16，7B→1000+ |
| `OS_MERGE_MAX_AT_ONCE` / `OS_MERGE_SEGMENTS_PER_TIER` / `OS_MERGE_FLOOR_SEGMENT` | `20` / `5` / `50mb` | 更激进合并段，减少每查询要遍历的 HNSW 图数 |
| `BATCH_SIZE` | `20000` | 大 batch 只拖慢本地去重，对 segment 无帮助 |
| `MSEARCH_CHUNK` | `1000` | 缩小到 500 可提高并行粒度（需配合 orjson） |
| `MSEARCH_WORKERS` | `20` | **按集群 vCPU 上调**（如 512 vCPU → 64–128）以打满集群 |
| `orjson` | 自动检测 | 安装后自动启用，search 序列化提速 5-10× |

> 以上除 `MSEARCH_WORKERS` / `MSEARCH_CHUNK` / `BATCH_SIZE` / `orjson` 外均为**索引级设置**，改后需重建索引才生效。

---

## 相关文档

- 架构级取舍与选型（pgvector vs OpenSearch，含官方引用）：见笔记《pgvector 边写边查去重场景调研》。
- 客户落地步骤：见《视频向量去重 · 开发实践指南》。
