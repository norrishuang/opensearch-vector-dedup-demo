# Vector Dedup — Test Demo（OpenSearch / RDS PostgreSQL）

验证"顺序批量向量去重"方案的测试程序，**同时支持两种向量引擎**：

- **Amazon OpenSearch Service**（Faiss HNSW kNN）
- **Amazon RDS for PostgreSQL + pgvector**（HNSW cosine）

生成模拟的 768 维视频 Embedding 向量，用 **numpy 矩阵乘法做本地去重**，再通过
"径向检索 → 批量写入 → 刷新"的顺序批量循环，把去重后的唯一向量写入所选引擎。
自带 **1 亿（100M）级别**压测能力与**去重准确率**评估。用 `--backend` 一键切换引擎，
其余流程与统计完全一致，方便横向对比两种引擎的吞吐与准确率。

> 方案背景：视频训练集去重 —— 把视频编码为 768 维 FP32 归一化向量，移除余弦相似度
> ≥ 0.95 的近重复。核心是"单 Worker 顺序批量"，用批次间的刷新等待消除并行方案的
> 刷新窗口竞态，从而无需事后全量清洗即可逼近 100% 准确率。

---

## 核心特性

- **流式数据生成**：不一次性占满内存，逐批生成归一化向量，支持 1 亿+ 规模。
- **可控近重复 + 真值标注**：每个向量带 `group_id`，同组即近重复，用于精确测算去重准确率（泄漏的重复数）。
- **numpy 本地去重**：`batch @ batch.T` 一次算出批内全部两两余弦相似度，精确且高效；大批量自动切换分块变体控制内存。
- **双引擎可插拔**：`--backend opensearch` 或 `--backend pgvector`，同一套流程与统计，方便横向对比。
- **顺序批量循环**：本地去重 → 径向检索（并行）→ 批量写入（并行）→ 刷新，完全对齐方案设计。
- **Dry-run 模式**：无需任何集群/数据库，用真值 oracle 验证整条流水线与准确率逻辑。
- **多种鉴权**：OpenSearch 支持 Basic Auth 与 AWS SigV4；PostgreSQL 支持标准连接 + SSL（RDS 默认）。

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
│   ├── pg_client.py        # PostgreSQL + pgvector 后端：建表 / HNSW / 检索 / 写入
│   └── dedup_runner.py     # 顺序批量主循环 + 后端工厂 + 吞吐/准确率统计
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

> 仅跑 dry-run（不连集群）时，只需要 `numpy`；连真实集群才需要 `opensearch-py`
> 等其余依赖。

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
  "batches": 10,
  "elapsed_s": 5.09,
  "throughput_vps": 19630.3,
  "ground_truth_unique": 69859,
  "leaked_duplicates": 0,
  "accuracy_pct": 100.0
}
```

- `indexed` == `ground_truth_unique` 且 `leaked_duplicates == 0` 说明去重逻辑正确。

### 3. 连接你的 OpenSearch 集群跑 1 亿压测

**Basic Auth（用户名/密码）：**

```bash
export OS_HOST=your-domain.us-west-2.es.amazonaws.com
export OS_PORT=443
export OS_USERNAME=admin
export OS_PASSWORD='your-password'
export OS_SHARDS=8          # 按集群规模调整

python run_dedup.py --total 100000000 --batch-size 20000
```

**AWS SigV4（托管 Amazon OpenSearch Service）：**

```bash
export OS_HOST=your-domain.us-west-2.es.amazonaws.com
export OS_USE_AWS_AUTH=true
export OS_AWS_REGION=us-west-2
export OS_AWS_SERVICE=es     # OpenSearch Serverless 用 aoss

python run_dedup.py --total 100000000 --batch-size 20000
```

> 程序启动时会 **删除并重建** 目标索引（默认 `video_vectors`），请勿指向生产索引。

### 4. 连接 RDS PostgreSQL（pgvector）跑压测

先确认 RDS 实例已启用 `vector` 扩展（程序会自动执行 `CREATE EXTENSION IF NOT EXISTS vector`，
需要账号有相应权限；RDS PostgreSQL 15+ 原生支持 pgvector）。

```bash
export BACKEND=pgvector
export PG_HOST=your-instance.xxxxx.us-west-2.rds.amazonaws.com
export PG_PORT=5432
export PG_DB=vectordb
export PG_USER=postgres
export PG_PASSWORD='your-password'
export PG_SSLMODE=require          # RDS 默认要求 SSL
export PG_MAINTENANCE_WORK_MEM=4GB # 建 HNSW 索引内存，越大越快（见调优章节）

python run_dedup.py --backend pgvector --total 100000000 --batch-size 20000
```

> 程序启动时会 **DROP 并重建** 目标表（默认 `video_vectors`）及其 HNSW cosine 索引，请勿指向生产表。

---

## 两种引擎对照

| 维度 | OpenSearch | PostgreSQL + pgvector |
|---|---|---|
| 向量索引 | Faiss HNSW | pgvector HNSW |
| 相似度 | innerproduct（`min_score=1.95`，需归一化） | cosine 距离 `<=>`（`dist <= 0.05`） |
| 一致性 | 近实时（写入后需 ~1s 刷新） | 事务一致（commit 后立即可见，无需等待） |
| 检索 | `_msearch` 并行分块 | 默认 `single`：逐条带绑定参数 kNN（走 HNSW 索引），线程池并行 |
| 写入 | `_bulk` 并行分块 | `execute_values` 批量 INSERT，线程池并行 |
| 建库参数 | m=16, ef_construction=128, ef_search=32（`OS_EF_SEARCH`） | m=16, ef_construction=128, ef_search=128（`PG_HNSW_*`） |
| 单查询并行 | 集群多 shard/节点并行 | 单实例单进程，靠客户端多连接并发 |

> 因 PostgreSQL 是事务一致的，pgvector 后端**不需要** OpenSearch 那样的批间刷新等待，
> 单批周期通常更短；但大规模下 HNSW 索引维护开销与 OpenSearch 分布式扩展性各有取舍，
> 正是本 demo 想帮你实测对比的点。

---

## pgvector 建索引内存调优（重要）

pgvector 的 **HNSW 索引构建对内存极其敏感**。构建时整个图会尽量放进
`maintenance_work_mem`；**一旦放不下就会 spill 到磁盘，构建速度会下降一个数量级甚至更多**。
本 demo 采用"先建空索引、随插入增量维护"的方式，插入阶段同样受此参数影响。

### 关键参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `maintenance_work_mem` | 建/维护索引可用内存 | 尽量大：单次构建建议 2–8 GB（本 demo 默认 `PG_MAINTENANCE_WORK_MEM=2GB`） |
| `max_parallel_maintenance_workers` | 并行建索引的 worker 数 | 多核实例可设 2–4，加速构建（`PG_MAX_PARALLEL_MAINT_WORKERS`） |
| `work_mem` | 查询排序/哈希内存 | 检索并发高时适当调大 |
| `shared_buffers` | 缓存热数据（含索引页） | 建议 ≈ 实例内存的 25% |

程序会在建索引前按 `PG_MAINTENANCE_WORK_MEM` / `PG_MAX_PARALLEL_MAINT_WORKERS`
**自动执行 `SET`（会话级）**，并打印实际生效值；权限不足时告警但不中断。

### RDS 上如何设置

- **会话级（本 demo 已自动做）**：`SET maintenance_work_mem = '4GB';` —— 立即生效，只影响当前连接，无需重启。
- **实例级（更彻底）**：通过 **RDS 参数组（Parameter Group）** 修改 `maintenance_work_mem`、
  `max_parallel_maintenance_workers`、`shared_buffers` 等，然后关联到实例。
  注意：`maintenance_work_mem` 在 RDS 参数组里单位是 **KB**，例如 `4GB` 填 `4194304`。
- **内存要留足**：`maintenance_work_mem × 并行 worker 数` 不能超过实例可用内存，否则可能 OOM。
  实例内存不足时，宁可调小并行度也别让它 spill。

### 经验值参考

| 数据规模 | 建议 `maintenance_work_mem` | 说明 |
|---|---|---|
| 百万级 | 1–2 GB | 一般够用 |
| 千万级 | 4–8 GB | 强烈建议调大，否则明显变慢 |
| 亿级 | 8 GB+ 且配合大内存实例 | 关注 spill 与整体内存预算；必要时分区/分表 |

> 小贴士：构建期间可用 `EXPLAIN` 或观察 `pg_stat_progress_create_index` 查看进度；
> 若发现构建异常慢，多半是 `maintenance_work_mem` 不足导致 spill。

---

## pgvector 检索性能排查（重要）

若发现 pgvector 去重比对很慢，按下面顺序排查——**先测量，再优化**：

### 1. 先确认检索是否真的走了 HNSW 索引

程序在 `pgvector` 后端启动时会自动打印一条检索 `EXPLAIN`：

```
[diag] search plan USES HNSW index:
       Limit ...
         ->  Index Scan using video_vectors_embedding_idx on video_vectors ...
```

- 若显示 **`does NOT use index (!)`** 或看到 `Seq Scan`，说明每次比对在**全表顺序扫描**，
  数据一多就会急剧变慢——这通常是 `lateral` 模式下 `ORDER BY` 关联外层向量、planner 放弃索引导致的。
- **解决**：使用默认的 `PG_QUERY_MODE=single`（每条向量一次带绑定参数的 kNN 查询，**保证走 HNSW 索引**）。
  `lateral` 模式仅用于 A/B 对比。

### 2. 看分阶段耗时，定位真正瓶颈

运行结束的结果 JSON 里有 `phase_seconds` / `phase_pct`，会告诉你时间到底花在
**检索（search）** 还是 **写入（index_write）**：

```json
"phase_pct": { "local_dedup": 5.0, "search": 78.0, "index_write": 15.0, "refresh": 2.0 }
```

- `search` 占大头 → 走索引优化（见上）+ 降 `ef_search` + 调并发。
- `index_write` 占大头 → "边写边查"下 HNSW **增量插入**随数据量增长变贵，属于 pgvector 的固有特性；
  可考虑分区表、或先全量导入再建索引（但那样就不是流式去重了）。

加 `--report-every-batch` 可**逐批**打印各阶段耗时，直观看出 search / write 是否随索引变大而变慢：

```bash
python run_dedup.py --backend pgvector --total 5000000 --batch-size 20000 --report-every-batch
```

```
  batch      1 | n=20,000 | local 0.45s  search 1.20s  write 0.80s  refresh 0.00s | batch 2.45s | idx_total       13,985
  batch     50 | n=20,000 | local 0.44s  search 3.10s  write 2.60s  refresh 0.00s | batch 6.14s | idx_total      690,102
```

> 若 `search`/`write` 明显随 `idx_total` 增长而上升，说明瓶颈是 HNSW 规模效应（检索图更深、插入更贵），
> 而非固定开销——这对判断 pgvector 在你数据量下的可扩展性很关键。

### 3. 降低 ef_search（去重不需要高精度 Top-K）

去重只需判断"最近邻是否 ≤ 阈值距离"，`ef_search` 默认已从 256 降到 **128**，可再降：

```bash
export PG_HNSW_EF_SEARCH=64   # 更快，召回略降，可用准确率统计验证
```

### 4. 并发按 CPU 核数走，别盲目调高

`MSEARCH_WORKERS`（默认 20）= 并发 DB 连接数 = 同时占用的 CPU 核数上限。
**pgvector 单条 kNN 查询在单个后端进程里跑**，所以：

```bash
# 合理值 ≈ 实例 vCPU 数（留 1–2 核给写入/系统）；16 核实例设 ~12
export MSEARCH_WORKERS=12
```

> 设过高（如 16 核开 50）不会更快，反而因抢 CPU + 连接竞争变慢，且不能超过 `max_connections`。

### 5. 与 OpenSearch 的本质差异

OpenSearch 的每个查询会被**分散到多个 shard/节点并行**；pgvector 单个查询只在**单实例单进程**里跑，
只能靠客户端多连接并发。大规模、高吞吐场景这是两者的架构性差异，也是本 demo 想帮你实测量化的点。

---

## 命令行参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--backend` | 向量引擎：`opensearch` 或 `pgvector` | opensearch |
| `--total` | 处理的向量总数 | 100,000,000（1 亿） |
| `--batch-size` | 每批向量数 | 20,000 |
| `--dup-ratio` | 注入近重复的比例 | 0.30 |
| `--dim` | 向量维度 | 768 |
| `--index` | 索引名 | video_vectors |
| `--dry-run` | 跳过 OpenSearch，用真值 oracle 验证 | 关闭 |
| `--seed` | 随机种子 | 42 |
| `--progress-every` | 每 N 批打印一次进度 | 10 |
| `--report-every-batch` | 每批打印分阶段耗时（local/search/write/refresh），观察是否随索引变大而变慢 | 关闭 |

## 环境变量

连接与调优参数集中在 `src/config.py`，可用环境变量覆盖：

| 变量 | 说明 | 默认 |
|---|---|---|
| `BACKEND` | 向量引擎：opensearch / pgvector | opensearch |
| `OS_HOST` / `OS_PORT` | OpenSearch 端点 | localhost / 443 |
| `OS_USERNAME` / `OS_PASSWORD` | Basic Auth 凭据 | 空 |
| `OS_USE_AWS_AUTH` | 是否用 SigV4 | false |
| `OS_AWS_REGION` / `OS_AWS_SERVICE` | SigV4 区域/服务（es 或 aoss） | us-west-2 / es |
| `OS_INDEX` | 索引名 | video_vectors |
| `OS_SHARDS` / `OS_REPLICAS` | 主分片 / 副本数 | 8 / 0 |
| `OS_EF_SEARCH` | HNSW 检索候选队列（去重 top-1 用小值更快，索引级，改后需重建索引） | 32 |
| `BATCH_SIZE` | 批大小 | 20000 |
| `MSEARCH_CHUNK` / `BULK_CHUNK` | 检索/写入分块大小 | 1000 / 5000 |
| `MSEARCH_WORKERS` / `BULK_WORKERS` | 检索/写入并行度 | 20 / 4 |
| `REFRESH_WAIT_S` | 每批后等待刷新秒数（仅 OpenSearch） | 1.0 |
| `NEAR_DUP_SIM` | 注入近重复的余弦相似度（越低越难） | 0.99 |
| `PG_HOST` / `PG_PORT` | RDS PostgreSQL 端点 | localhost / 5432 |
| `PG_DB` / `PG_USER` / `PG_PASSWORD` | 数据库 / 用户 / 密码 | vectordb / postgres / 空 |
| `PG_SSLMODE` | SSL 模式（RDS 建议 require） | require |
| `PG_TABLE` | 表名 | video_vectors |
| `PG_HNSW_M` / `PG_HNSW_EF_CONSTRUCTION` / `PG_HNSW_EF_SEARCH` | pgvector HNSW 参数 | 16 / 128 / 128 |
| `PG_QUERY_MODE` | 检索模式：`single`（走 HNSW 索引，默认）/ `lateral`（批量单次往返，可能不走索引） | single |
| `PG_MAINTENANCE_WORK_MEM` | 建 HNSW 索引的内存（越大越快，见调优） | 2GB |
| `PG_MAX_PARALLEL_MAINT_WORKERS` | 建索引并行 worker 数（0=服务器默认） | 0 |

---

## 关键参数（对齐方案设计）

| 参数 | 值 | 说明 |
|---|---|---|
| 维度 | 768 (FP32) | 视频 Embedding，需 **L2 归一化** |
| 空间类型 | `innerproduct` | 归一化后等价余弦 |
| 去重阈值 | 余弦 ≥ 0.95 | `min_score = 1.95`（分数 = 1 + 余弦） |
| HNSW | m=16, ef_c=128, ef_search=32（可调 `OS_EF_SEARCH`） | Faiss 引擎 |
| 批大小 | 20,000 | 单周期 |
| 分块 | `_msearch` 1K/块、`_bulk` 5K/块 | 规避 100MB 请求上限 |

---

## 工作原理

每个批次依次执行（**批次内并行、批次间顺序**）：

1. **本地去重**：`batch @ batch.T` 求批内两两余弦，丢弃 ≥ 0.95 的重复（精确）。
2. **`_msearch` 径向检索**：把幸存向量对已建索引做检索，命中（余弦 ≥ 0.95）即为重复。
3. **`_bulk` 写入**：仅写入未命中的向量。
4. **等待刷新**：`sleep(1s)`，使本批写入对下一批可见。

批次间的顺序 + 刷新等待，确保每次检索都能看到此前写入的全部向量，消除并行方案
的刷新窗口竞态导致的重复泄漏。详见方案设计文档。

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

## License

[MIT](LICENSE)
