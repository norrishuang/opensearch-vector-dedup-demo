# Vector Dedup — Test Demo（Amazon OpenSearch Service）

验证"顺序批量向量去重"方案的测试程序。生成模拟的 768 维视频 Embedding 向量，
用 **numpy 矩阵乘法做本地去重**，再通过"径向检索 → 批量写入 → 刷新"的顺序批量循环，
把去重后的唯一向量写入 **Amazon OpenSearch Service**（Faiss HNSW kNN）。
自带 **1 亿（100M）级别**压测能力与**去重准确率**评估。

> 方案背景：视频训练集去重 —— 把视频编码为 768 维 FP32 归一化向量，移除余弦相似度
> ≥ 0.95 的近重复。核心是"单 Worker 顺序批量"，用批次间的显式刷新消除并行方案的
> 刷新窗口竞态，从而无需事后全量清洗即可逼近 100% 准确率。
>
> 另提供一个**实验性的 pgvector（RDS PostgreSQL）后端**用于横向对比，见文末
> [附录](#附录pgvector-后端实验性)。压测显示 pgvector 在本"边写边查、索引持续增长"
> 场景下性能不佳，仅作对比参考。

---

## 📖 调优必读

**OpenSearch kNN 边写边查的调优要点、实测数据趋势与大规模架构建议，见
[docs/opensearch-tuning.md](docs/opensearch-tuning.md)。** 本项目默认值已按其中的最佳实践设置。

---

## 核心特性

- **流式数据生成**：不一次性占满内存，逐批生成归一化向量，支持 1 亿+ 规模。
- **可控近重复 + 真值标注**：每个向量带 `group_id`，同组即近重复，用于精确测算去重准确率（泄漏的重复数）。
- **numpy 本地去重**：`batch @ batch.T` 一次算出批内全部两两余弦相似度，精确且高效；大批量自动切换分块变体控制内存。
- **顺序批量循环**：本地去重 → `_msearch` 径向检索（并行）→ `_bulk` 写入（并行）→ 显式刷新。
- **分阶段计时**：`--report-every-batch` 逐批打印 local/search/write/refresh 耗时，定位瓶颈。
- **Dry-run 模式**：无需任何集群，用真值 oracle 验证整条流水线与准确率逻辑。
- **多种鉴权**：Basic Auth 与 AWS SigV4（托管 Amazon OpenSearch Service / Serverless）。
- **可插拔后端**：主用 OpenSearch；附带实验性 pgvector 后端用于对比（`--backend pgvector`）。

---

## 目录结构

```
opensearch-vector-dedup-demo/
├── run_dedup.py            # CLI 入口
├── src/
│   ├── config.py           # 所有可调参数（支持环境变量 / CLI 覆盖）
│   ├── data_generator.py   # 流式模拟向量生成 + 真值 group_id
│   ├── local_dedup.py      # numpy 矩阵乘法本地去重（含分块变体）
│   ├── backend_base.py     # 向量后端统一接口（VectorBackend）
│   ├── os_client.py        # OpenSearch 后端：建索引 / _msearch / _bulk / refresh
│   ├── pg_client.py        # pgvector 后端（实验性）
│   └── dedup_runner.py     # 顺序批量主循环 + 后端工厂 + 吞吐/准确率统计
├── docs/
│   └── opensearch-tuning.md  # OpenSearch 调优最佳实践（必读）
├── tests/test_dedup.py     # 单元测试（本地去重正确性 + dry-run 零泄漏）
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
cd opensearch-vector-dedup-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 仅跑 dry-run（不连集群）时只需 `numpy`；连真实集群才需要 `opensearch-py` 等其余依赖。

### 2. 先跑 Dry-run（无需集群，验证流水线）

```bash
python run_dedup.py --dry-run --total 100000 --batch-size 10000
```

输出示例：

```json
{
  "processed": 100000,
  "indexed": 69859,
  "local_discarded": 7508,
  "os_discarded": 22633,
  "throughput_vps": 19630.3,
  "ground_truth_unique": 69859,
  "leaked_duplicates": 0,
  "accuracy_pct": 100.0
}
```

- `indexed == ground_truth_unique` 且 `leaked_duplicates == 0` 说明去重逻辑正确。

### 3. 连接 OpenSearch 集群压测

**Basic Auth（用户名/密码）：**

```bash
export OS_HOST=your-domain.us-west-2.es.amazonaws.com
export OS_PORT=443
export OS_USERNAME=admin
export OS_PASSWORD='your-password'
export OS_SHARDS=2                # 按数据量调整，见调优文档
export MSEARCH_WORKERS=64         # 按集群 vCPU 上调以打满集群

python run_dedup.py --total 100000000 --batch-size 20000 --report-every-batch
```

**AWS SigV4（托管 Amazon OpenSearch Service）：**

```bash
export OS_HOST=your-domain.us-west-2.es.amazonaws.com
export OS_USE_AWS_AUTH=true
export OS_AWS_REGION=us-west-2
export OS_AWS_SERVICE=es          # OpenSearch Serverless 用 aoss

python run_dedup.py --total 100000000 --batch-size 20000 --report-every-batch
```

> 程序启动时会 **删除并重建** 目标索引（默认 `video_vectors`），请勿指向生产索引。
> 调参前请先读 [docs/opensearch-tuning.md](docs/opensearch-tuning.md)。

---

## 工作原理

每个批次依次执行（**批次内并行、批次间顺序**）：

1. **本地去重**：`batch @ batch.T` 求批内两两余弦，丢弃 ≥ 0.95 的重复（精确）。
2. **`_msearch` 径向检索**：把幸存向量对已建索引做检索，命中（余弦 ≥ 0.95）即为重复。
3. **`_bulk` 写入**：仅写入未命中的向量。
4. **显式刷新**：调用同步 `_refresh`，使本批写入对下一批可见（`refresh_interval=-1` 下这是唯一可见性来源）。

批次间顺序 + 显式刷新，确保每次检索都能看到此前写入的全部向量，消除并行方案的
刷新窗口竞态导致的重复泄漏。

---

## 关键参数（对齐方案设计）

| 参数 | 值 | 说明 |
|---|---|---|
| 维度 | 768 (FP32) | 视频 Embedding，需 **L2 归一化** |
| 空间类型 | `innerproduct` | 归一化后等价余弦 |
| 去重阈值 | 余弦 ≥ 0.95 | `min_score = 1.95`（分数 = 1 + 余弦） |
| HNSW | m=16, ef_c=128, ef_search=32（可调 `OS_EF_SEARCH`） | Faiss 引擎 |
| 批大小 | 20,000 | 单周期（大 batch 只拖慢本地去重） |
| 分块 | `_msearch` 1K/块、`_bulk` 5K/块 | 规避 100MB 请求上限 |

---

## 命令行参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--backend` | 向量引擎：`opensearch`（默认）或 `pgvector`（实验性） | opensearch |
| `--total` | 处理的向量总数 | 100,000,000（1 亿） |
| `--batch-size` | 每批向量数 | 20,000 |
| `--dup-ratio` | 注入近重复的比例 | 0.30 |
| `--dim` | 向量维度 | 768 |
| `--index` | 索引名 | video_vectors |
| `--dry-run` | 跳过集群，用真值 oracle 验证 | 关闭 |
| `--seed` | 随机种子 | 42 |
| `--progress-every` | 每 N 批打印一次进度 | 10 |
| `--report-every-batch` | 每批打印分阶段耗时（local/search/write/refresh） | 关闭 |

## 环境变量（OpenSearch）

连接与调优参数集中在 `src/config.py`，可用环境变量覆盖。**各项含义与推荐值见
[docs/opensearch-tuning.md](docs/opensearch-tuning.md)。**

| 变量 | 说明 | 默认 |
|---|---|---|
| `OS_HOST` / `OS_PORT` | OpenSearch 端点 | localhost / 443 |
| `OS_USERNAME` / `OS_PASSWORD` | Basic Auth 凭据 | 空 |
| `OS_USE_AWS_AUTH` | 是否用 SigV4 | false |
| `OS_AWS_REGION` / `OS_AWS_SERVICE` | SigV4 区域/服务（es 或 aoss） | us-west-2 / es |
| `OS_INDEX` | 索引名 | video_vectors |
| `OS_SHARDS` / `OS_REPLICAS` | 主分片 / 副本数 | 8 / 0 |
| `OS_EF_SEARCH` | HNSW 检索候选队列（去重 top-1 用小值更快，索引级） | 32 |
| `OS_REFRESH_INTERVAL` | 自动刷新周期。`-1` 禁用，靠每批显式 refresh | -1 |
| `OS_MERGE_MAX_AT_ONCE` / `OS_MERGE_SEGMENTS_PER_TIER` / `OS_MERGE_FLOOR_SEGMENT` | 段合并策略（更激进合并，索引级） | 20 / 5 / 50mb |
| `BATCH_SIZE` | 批大小 | 20000 |
| `MSEARCH_CHUNK` / `BULK_CHUNK` | 检索/写入分块大小 | 1000 / 5000 |
| `MSEARCH_WORKERS` / `BULK_WORKERS` | 检索/写入并行度（按集群 vCPU 上调 workers） | 20 / 4 |
| `REFRESH_WAIT_S` | 每批后额外 sleep 秒数（显式 refresh 已同步，通常 0） | 0 |
| `NEAR_DUP_SIM` | 注入近重复的余弦相似度（越低越难） | 0.99 |

---

## 运行测试

```bash
python tests/test_dedup.py
# 或
python -m pytest tests/ -q
```

覆盖：本地去重正确性、稠密/分块变体一致性、生成器真值组数、dry-run 零泄漏。

---

## 说明

- 模拟数据是**随机向量 + 注入近重复**，非真实视频 Embedding；用于验证去重/写入
  流水线与吞吐，不代表真实数据分布下的 HNSW 召回。
- 真实场景请把 `data_generator` 换成你的向量来源（S3 / SQS 等），其余流程不变。
- 一次性压测任务建议 `OS_REPLICAS=0`，并把 `knn.memory.circuit_breaker.limit`
  调到 75%（程序会尝试自动设置）。

---

## 附录：pgvector 后端（实验性）

> ⚠️ **仅作横向对比参考。** 压测显示 pgvector 在本"边写边查、索引持续增长"场景下
> 性能不佳——单实例无法水平分片、单查询单核、增量插入随图增长变贵。适用规模
> 大致在**低千万级、能装进单机内存**；亿级 / 十亿级请用 OpenSearch。详细原理与
> 取舍见笔记《pgvector 边写边查去重场景调研》。

### 运行

先确认 RDS 实例支持 `vector` 扩展（程序会自动 `CREATE EXTENSION IF NOT EXISTS vector`，
RDS PostgreSQL 15+ 原生支持）。

```bash
export BACKEND=pgvector
export PG_HOST=your-instance.xxxxx.us-west-2.rds.amazonaws.com
export PG_PORT=5432
export PG_DB=vectordb
export PG_USER=postgres
export PG_PASSWORD='your-password'
export PG_SSLMODE=require            # RDS 默认要求 SSL
export PG_MAINTENANCE_WORK_MEM=4GB   # 建 HNSW 索引内存，越大越快

python run_dedup.py --backend pgvector --total 5000000 --batch-size 20000 --report-every-batch
```

> 程序启动时会 **DROP 并重建** 目标表及其 HNSW cosine 索引，请勿指向生产表。
> 启动时会打印一条 `EXPLAIN` 用于确认检索走了 HNSW 索引（而非 Seq Scan）。

### 两种引擎对照

| 维度 | OpenSearch | PostgreSQL + pgvector |
|---|---|---|
| 向量索引 | Faiss HNSW | pgvector HNSW |
| 相似度 | innerproduct（`min_score=1.95`，需归一化） | cosine 距离 `<=>`（`dist <= 0.05`） |
| 一致性 | 近实时（写入后需显式 refresh） | 事务一致（commit 后立即可见） |
| 检索 | `_msearch` fan-out 到多分片并行 | 默认 `single`：逐条绑定参数 kNN（走 HNSW 索引），线程池并发 |
| 写入 | `_bulk` 并行分块 | `execute_values` 批量 INSERT，线程池并发 |
| 单查询并行 | 集群多 shard/节点并行 | 单实例单进程，靠客户端多连接并发 |
| 水平扩展 | 加节点，近线性 | 不能分片单个 HNSW 图 |
| 适用规模 | 亿级 ~ 十亿级 | 低千万级（单机内存内） |

### pgvector 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `PG_HOST` / `PG_PORT` | RDS 端点 | localhost / 5432 |
| `PG_DB` / `PG_USER` / `PG_PASSWORD` | 数据库 / 用户 / 密码 | vectordb / postgres / 空 |
| `PG_SSLMODE` | SSL 模式（RDS 建议 require） | require |
| `PG_TABLE` | 表名 | video_vectors |
| `PG_HNSW_M` / `PG_HNSW_EF_CONSTRUCTION` / `PG_HNSW_EF_SEARCH` | HNSW 参数 | 16 / 128 / 128 |
| `PG_QUERY_MODE` | `single`（走索引，默认）/ `lateral`（批量单次往返，可能不走索引） | single |
| `PG_MAINTENANCE_WORK_MEM` | 建 HNSW 索引内存（越大越快，放不下会 spill 到磁盘极慢） | 2GB |
| `PG_MAX_PARALLEL_MAINT_WORKERS` | 建索引并行 worker 数（0=服务器默认） | 0 |

### pgvector 排查要点

- **检索是否走 HNSW 索引**：看启动时的 `EXPLAIN`，出现 `Seq Scan` 说明没走索引（多为 `lateral` 模式的关联 ORDER BY 导致），用默认 `single` 模式。
- **建索引内存**：`maintenance_work_mem` 放不下图就 spill 到磁盘、慢一个数量级；千万级建议 4–8GB。RDS 参数组里单位是 KB（`4GB` 填 `4194304`）。
- **并发按 CPU 核数**：`MSEARCH_WORKERS` = 并发连接数 = 占用核数上限；pgvector 单查询单核，设过高（超 vCPU / `max_connections`）反而更慢。
- **降 ef_search**：去重 top-1，`PG_HNSW_EF_SEARCH=64` 更快，用准确率验证召回。

---

## License

[MIT](LICENSE)
