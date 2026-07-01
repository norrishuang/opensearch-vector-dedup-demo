# OpenSearch Vector Dedup — Test Demo

在 **Amazon OpenSearch Service** 上验证"顺序批量向量去重"方案的测试程序。
生成模拟的 768 维视频 Embedding 向量，用 **numpy 矩阵乘法做本地去重**，再通过
`_msearch` 径向检索 + `_bulk` 写入的顺序批量循环，把去重后的唯一向量写入 OpenSearch。
自带 **1 亿（100M）级别**压测能力与**去重准确率**评估。

> 方案背景：视频训练集去重 —— 把视频编码为 768 维 FP32 归一化向量，移除余弦相似度
> ≥ 0.95 的近重复。核心是"单 Worker 顺序批量"，用批次间的刷新等待消除并行方案的
> 刷新窗口竞态，从而无需事后全量清洗即可逼近 100% 准确率。

---

## 核心特性

- **流式数据生成**：不一次性占满内存，逐批生成归一化向量，支持 1 亿+ 规模。
- **可控近重复 + 真值标注**：每个向量带 `group_id`，同组即近重复，用于精确测算去重准确率（泄漏的重复数）。
- **numpy 本地去重**：`batch @ batch.T` 一次算出批内全部两两余弦相似度，精确且高效；大批量自动切换分块变体控制内存。
- **顺序批量循环**：本地去重 → `_msearch`（20×1K 并行）→ `_bulk`（4×5K 并行）→ 等待刷新，完全对齐方案设计。
- **Dry-run 模式**：无需 OpenSearch 集群，用真值 oracle 验证整条流水线与准确率逻辑。
- **两种鉴权**：Basic Auth（自建/细粒度访问控制）与 AWS SigV4（托管 Amazon OpenSearch Service / Serverless）。

---

## 目录结构

```
opensearch-vector-dedup-demo/
├── run_dedup.py            # CLI 入口
├── src/
│   ├── config.py           # 所有可调参数（支持环境变量 / CLI 覆盖）
│   ├── data_generator.py   # 流式模拟向量生成 + 真值 group_id
│   ├── local_dedup.py      # numpy 矩阵乘法本地去重（含分块变体）
│   ├── os_client.py        # OpenSearch 客户端：建索引 / _msearch / _bulk / refresh
│   └── dedup_runner.py     # 顺序批量主循环 + 吞吐/准确率统计
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

---

## 命令行参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--total` | 处理的向量总数 | 100,000,000（1 亿） |
| `--batch-size` | 每批向量数 | 20,000 |
| `--dup-ratio` | 注入近重复的比例 | 0.30 |
| `--dim` | 向量维度 | 768 |
| `--index` | 索引名 | video_vectors |
| `--dry-run` | 跳过 OpenSearch，用真值 oracle 验证 | 关闭 |
| `--seed` | 随机种子 | 42 |
| `--progress-every` | 每 N 批打印一次进度 | 10 |

## 环境变量

连接与调优参数集中在 `src/config.py`，可用环境变量覆盖：

| 变量 | 说明 | 默认 |
|---|---|---|
| `OS_HOST` / `OS_PORT` | OpenSearch 端点 | localhost / 443 |
| `OS_USERNAME` / `OS_PASSWORD` | Basic Auth 凭据 | 空 |
| `OS_USE_AWS_AUTH` | 是否用 SigV4 | false |
| `OS_AWS_REGION` / `OS_AWS_SERVICE` | SigV4 区域/服务（es 或 aoss） | us-west-2 / es |
| `OS_INDEX` | 索引名 | video_vectors |
| `OS_SHARDS` / `OS_REPLICAS` | 主分片 / 副本数 | 8 / 0 |
| `BATCH_SIZE` | 批大小 | 20000 |
| `MSEARCH_CHUNK` / `BULK_CHUNK` | 检索/写入分块大小 | 1000 / 5000 |
| `MSEARCH_WORKERS` / `BULK_WORKERS` | 检索/写入并行度 | 20 / 4 |
| `REFRESH_WAIT_S` | 每批后等待刷新秒数 | 1.0 |
| `NEAR_DUP_SIM` | 注入近重复的余弦相似度（越低越难） | 0.99 |

---

## 关键参数（对齐方案设计）

| 参数 | 值 | 说明 |
|---|---|---|
| 维度 | 768 (FP32) | 视频 Embedding，需 **L2 归一化** |
| 空间类型 | `innerproduct` | 归一化后等价余弦 |
| 去重阈值 | 余弦 ≥ 0.95 | `min_score = 1.95`（分数 = 1 + 余弦） |
| HNSW | m=16, ef_c=128, ef_search=256 | Faiss 引擎 |
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
