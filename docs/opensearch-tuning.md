# OpenSearch kNN 边写边查去重 · 调优最佳实践

> 面向本项目的去重场景：**边写边查、索引持续增长**——每来一批向量，先对现有索引做
> kNN 检索判重，没命中才写入。
>
> 本项目 `src/config.py` 的默认值已按下述最佳实践设置。

> **测试环境说明**：本文的参数建议基于以下集群配置压测验证：
> - **集群规模**：16 × `r8g.8xlarge`（每节点 32 vCPU / 256 GB 内存，共 512 vCPU）
> - **索引配置**：16 主分片 / 0 副本
> - **向量**：768 维 FP32，L2 归一化，去重阈值余弦 ≥ 0.95
> - **数据规模**：10M 向量压测
>
> 未在更大规模集群上验证过，请按你自己的集群规模和数据量对参数做等比例调整和验证。

---

## 1. 关闭周期性自动刷新，只靠每批显式 refresh

**参数**：
```bash
export OS_REFRESH_INTERVAL=-1   # 默认值
export REFRESH_WAIT_S=0         # 默认值
```

**好处**：`refresh_interval` 默认 `1s` 时，一批几十秒的 `search`+`write` 期间，后台会
触发几十次自动刷新，每次刷新都会切出一个新 segment，segment 越多、`search` 越慢。
本项目在每批写入后调用**同步的** `index.refresh()`，可见性完全由它保证——下一批一定
看得到上一批。禁用自动刷新可以避免中途产生对正确性无贡献的多余 segment。显式
`index.refresh()` 是同步调用（返回即刷新完成），无需额外 sleep。

---

## 2. 去重只需 top-1，用小的 ef_search

**参数**：
```bash
export OS_EF_SEARCH=32   # 默认值；漏检偏多可调到 64
```

**好处**：去重只需判断"最近 1 个邻居是否 ≥ 0.95"，不需要高召回 Top-K 排序，`ef_search`
调小可以显著降低每次查询的 CPU 开销。

> ⚠️ `ef_search` 是索引级设置，仅在创建索引时生效，改动后需重建索引。

---

## 3. 启用 memory-optimized search，配合默认段合并策略

**参数**：
```bash
export OS_KNN_MEMORY_OPTIMIZED_SEARCH=true   # 需 OpenSearch 3.1+
# 段合并策略保持 OpenSearch 默认值，不需要额外调整
```

**好处**：每批显式 refresh 会持续产生新 segment，段越多，每次 kNN 查询要遍历的独立
HNSW 图越多。`memory_optimized_search` 让 Faiss 引擎通过内存映射（mmap）+ 操作系统
页缓存来访问向量索引文件，而不是把整个索引全量加载到 off-heap 内存，使查询性能
不再随 segment 数量增长而明显下降。实测：启用后 `search` 阶段随索引从 0 增长到 200
万+ 行几乎保持恒定（2.6-3.1s），不再随规模上升。

**段合并策略保持默认即可**：由于 search 已不受 segment 数量影响，无需再通过调大
`merge.scheduler.max_thread_count`、调小 `merge.policy.max_merged_segment` 等参数
让合并更激进。保持默认合并参数反而更稳定——过于激进的合并目标会让后台合并线程
跟不上写入速度，触发 OpenSearch 的磁盘 IO 限流（auto throttle），拖慢同步
`index.refresh()` 的返回时间。实测：默认合并参数下，`refresh` 稳定在 0.2-0.9s，
连续 60+ 批次没有出现阻塞尖峰。

> ⚠️ `memory_optimized_search` 是索引级设置，改动后需重建索引；仅支持 Faiss + HNSW
> 引擎，且要求索引在 OpenSearch 2.19+ 创建。

**batch 大小建议**：推荐 `BATCH_SIZE=20000-40000`。batch 越大本地去重（numpy 矩阵
运算）越慢，需要结合客户端算力权衡。

---

## 4. 客户端 JSON 序列化用 orjson 替代标准库

**参数**：
```bash
pip install orjson>=3.9.0   # 安装后代码自动检测启用
export MSEARCH_CHUNK=500    # 配合 orjson 可以调小，提高并行粒度
export MSEARCH_WORKERS=64   # 按集群 vCPU 规模上调
```

**好处**：`_msearch` 请求体需要把每条 768 维向量序列化为 JSON，标准库 `json.dumps`
是纯 Python 实现，序列化大量浮点数组时成为客户端瓶颈。`orjson` 是 C/Rust 实现，
序列化速度是标准库的 5-10 倍，且执行期间会释放 GIL，配合多线程并发可以显著提升
吞吐。代码已集成自动检测：安装了 `orjson` 就自动使用，否则回退到标准 `json`。

---

## 5. 检索并发要匹配集群规模

**参数**：
```bash
export MSEARCH_WORKERS=64    # 按集群 vCPU 数量级调整
export BULK_WORKERS=8
export BULK_CHUNK=5000
```

**好处**：`MSEARCH_WORKERS` 控制并发检索连接数。设置过低（如默认的 20）会让大规模
集群的算力大部分闲置，无法充分利用集群资源。按集群总 vCPU 数量级设置并发数
（如 512 vCPU 集群设 64 左右），可以让请求更充分地打到各个分片上。

---

## 6. 参数汇总

| 参数 | 建议值 | 好处 |
|---|---|---|
| `OS_REFRESH_INTERVAL` | `-1` | 禁用自动刷新，避免产生多余 segment |
| `REFRESH_WAIT_S` | `0` | 显式 refresh 已同步，无需额外等待 |
| `OS_EF_SEARCH` | `32` | 去重只需 top-1，降低每次查询的 CPU 开销 |
| `OS_KNN_MEMORY_OPTIMIZED_SEARCH` | `true`（需 3.1+） | mmap 访问索引，search 不再随 segment 数增长而变慢 |
| 段合并相关参数 | 保持 OpenSearch 默认值 | 过激进的合并目标会触发磁盘 IO 限流，拖慢 refresh |
| `BATCH_SIZE` | `20000-40000` | 按客户端算力权衡；越大本地去重越慢 |
| `orjson` | 安装后自动启用 | 客户端序列化提速 5-10× |
| `MSEARCH_CHUNK` | `500` | 配合 orjson 提高并行粒度 |
| `MSEARCH_WORKERS` | `64`（按集群 vCPU 调整） | 打满集群检索并发 |
| `BULK_CHUNK` / `BULK_WORKERS` | `5000` / `8` | 并行写入吞吐 |

> `OS_EF_SEARCH` / `OS_KNN_MEMORY_OPTIMIZED_SEARCH` 为索引级设置，改动后需重建索引才生效。
> `MSEARCH_CHUNK` / `MSEARCH_WORKERS` / `BULK_CHUNK` / `BULK_WORKERS` / `BATCH_SIZE` /
> `orjson` 为客户端/运行时参数，可随时调整无需重建索引。
