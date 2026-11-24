# 片库镜头检索向量服务（共享素材版）

用可播镜头实时维护视频检索向量：导入按时间排序的 1～100,000 个镜头（每个带 16 维特征、注意力分数），
按单号登记/撤销镜头隐藏区间，每次变更后以 **O(log N)** 返回由可播镜头算出的检索向量
（注意力分数 softmax 加权特征和）。

同一部片分渠道上线时，各渠道用不同片号（`video_id`）导入**完全一致的镜头**。服务按内容指纹识别，
让这些片号**共用一份基础检索数据（catalog）**：叶子特征/分数与整棵线段树只常驻一份，
每个片号只额外持有自己的稀疏限播订单（overlay）。各片号独立接单/撤单，互不影响。

## 运行

```bash
# Docker Compose
docker compose up --build

# 或本地运行
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务监听 `:8000`，健康检查 `GET /health`，交互文档 `GET /docs`。

演示：一部片 3 个渠道片号共享索引、各自限播、片号迁走后索引释放、共享 vs 独立副本的常驻空间对比：

```bash
bash scripts/demo.sh            # BASE=http://localhost:8000 可覆盖
```

基准（N=100,000 的索引/overlay 空间、V 个片号的共享收益、精度、单次接/撤单耗时）：

```bash
python scripts/bench_storage.py
```

自测：

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/videos/{video_id}/shots` | 导入镜头。body: `{"shots":[{"feature":[16 个 −1~1],"score":−1000~1000}, ...]}`，1～100,000 条。内容与已有 catalog 完全一致时直接复用、不再建树；否则整体替换该片号镜头并清空该片号全部订单 |
| `POST` | `/videos/{video_id}/restrictions` | 登记隐藏单。body: `{"order_id":"lic-A","start":2,"end":6}`，区间 0 起算、左闭右开 |
| `DELETE` | `/videos/{video_id}/restrictions/{order_id}` | 撤销一单；仍被其他单覆盖的镜头保持隐藏 |
| `GET` | `/videos/{video_id}/retrieval-vector` | 查询当前检索向量 |
| `GET` | `/storage` | 常驻索引空间：共享 catalog、各片号 overlay，以及"每个片号各存一份"的对照值 |
| `GET` | `/health` | 健康检查 |

统一响应（`/storage` 除外）：

```json
{"video_id":"channel-a","shot_count":10,"visible_count":6,"status":"ok",
 "vector":[...16 维...],"catalog_id":"ab83886ba70cdf71","shared_videos":3}
```

- `catalog_id`：内容指纹（特征+分数的 SHA-256 前 16 位）。相同内容 → 相同 id → 同一份索引。
- `shared_videos`：当前共用这份索引的片号数。
- **幂等**：同片号、同 `order_id`、同范围重送 → 200，状态不变。
- **冲突**：同片号、同 `order_id`、不同范围 → 409（提示先撤销）。
- **订单隔离**：`order_id` 只在单个片号内唯一。同一 `order_id` 在不同片号可登记不同区间，互不影响。
- **全隐藏**：`status="withdrawn"`、`vector=null` —— 即撤下索引的标记；撤销后自动恢复。
- 校验失败 422（维度数、特征/分数越界、镜头数越界），区间语义错误 400（`start>=end`、`end>N`），未知片号/单号 404。

## 实现

### 共享索引 + 稀疏 overlay（`app/segment_tree.py`）

`ShotIndex`：**按内容共享、只读**的基础索引。叶子 payload（特征/分数）与整棵 softmax 线段树，
节点存 `cnt / mx / sm / vec[16]`，兄弟合并用 online-softmax 平移：

```
m   = max(mx_l, mx_r)
sm  = sm_l·exp(mx_l−m) + sm_r·exp(mx_r−m)
vec = vec_l·exp(mx_l−m) + vec_r·exp(mx_r−m)
```

`RestrictionIndex`：每个片号一个，仅保存**被订单改写过**的节点：

- `covers[node]`：覆盖该规范节点的生效单单数（懒引用计数，0 的条目直接删掉）
- `overrides[node]`：该节点的可见聚合（覆盖态或子节点被改写时）；`_EMPTY` 表示整段隐藏；
  聚合恰好等于基础索引时删除，保证 overlay 只与生效单数量 R 相关
- `orders[order_id]`：该片号的区间

接单对 O(log N) 个规范节点 +1 并自底向上重算；撤单 -1 同理。`covers>0` 的节点贡献为空，
多单重叠时同一镜头计数 ≥1，撤销一单后若仍被其他单覆盖则保持隐藏；计数归零的节点若子节点也未改写
则回落到共享基础索引、不占空间。查询 = 根聚合的 `vec / sm`，基础索引没有 overlay 时直接读共享根。

### 内容寻址与生命周期（`app/main.py`）

`_CATALOGS[指纹] = (ShotIndex, {片号})`，`_VIDEOS[片号] = (指纹, ShotIndex, RestrictionIndex)`。

- **导入**：先校验并算出指纹；若已有同指纹 catalog，**零建树**直接挂接。
  同一内容正在建树时，并发导入等待该次构建完成再共享（有界等待 120 s），因此**任何时刻只驻留一份索引**，
  且并发导入不会各建一棵树；若构建线程异常，其等待者会自行建树后继续。
- **并发导入**：建树 O(N) 在锁外进行，锁内只做指针替换。建树期间旧索引照常可查、可接/撤单。
- **成功**：原子换到新 catalog 后，该片号的旧订单被清空（新镜头列表与旧区间无关）。
- **失败**：校验失败（422）在建树之前；建树抛错时尚未替换任何指针，原镜头与原订单继续生效。
- **释放**：片号迁走后从旧 catalog 的引用集合移除；最后一个片号离开时删除 catalog，
  旧索引（叶子 + 整棵树）随引用消失而释放。

### 复杂度

- 接单 / 撤单：O(log N) 次节点访问，每次 16 维常数合并；查询 O(1)（根聚合）。
- 空间：V 个片号共用 N 个镜头、合计 R 张生效单 → `O(N + V + R log N)`。
  每个 overlay 只含有单覆盖或被其子节点影响的 O(R log N) 个节点，空 overlay 约百字节。

## 实测结果

环境：本机 Python 3.11.2，pytest 9.2.0，fastapi 0.165.0。

- **62 个测试全部通过**（`tests/test_service.py` + `tests/test_shared_index.py`）：
  - 40 组随机用例（N=1…1000，每步接/撤单后与参考实现比对）：逐维误差 ≤ 1e-9 ✅
  - 极端值（分数 ±1000、特征 ±1）、N=100,000 大样本、全隐藏→恢复、撤空后 overlay 归零 ✅
  - 共享：3 个片号同内容 → 1 个 catalog；同 `order_id` 在 a/b 不同区间互不影响；不同内容各自 catalog ✅
  - 生命周期：b 迁走 → catalog 仍服务 a；a 也迁走 → catalog 释放（`weakref` 验证 ShotIndex 真被回收）✅
  - 重导同内容 → 仍复用同一 catalog，但该片号订单被清空 ✅
  - 重导失败（422 与模拟建树抛错 500）→ 原镜头、原订单、原向量不变 ✅
  - 建树期间（锁外）另一线程查向量、接新单均成功且 <1 s ✅
  - 8 线程并发导入同内容 → **只建 1 棵树**（计数断言），1 个 catalog，向量一致 ✅
  - 建树线程失败时等待中的并发导入不会卡死，会自行建树成功 ✅
  - 读线程持续轮询、同时反复重导 → 只看到完整旧/新状态，无混合态 ✅
  - API：重叠撤销、幂等重送、409、404、422/400、withdrawn 标记 ✅
- **精度实测**（`scripts/bench_storage.py`，N=100,000 + 200 张生效单，可见 6,001 镜头）：
  逐维最大误差 **1.665e-16**（阈值 1e-9）。
- **常驻索引空间实测**（N=100,000，`tracemalloc` 对照）：

  | 项 | 值 |
  |---|---|
  | 一份共享索引 `byte_size()`（含 Python 对象开销） | **55,541,660 B ≈ 53.0 MiB** |
  | 同一索引 `tracemalloc` 实际分配 | 41,091,720 B ≈ 39.2 MiB |
  | overlay，R=0 单 | 192 B |
  | overlay，R=40 单 | 216,618 B（covers 335 / overrides 833） |
  | overlay，R=200 单 | 849,658 B（covers 1,406 / overrides 3,130） |

  V 个片号同内容时（对照"每片号各存一份索引"）：

  | V | 旧布局（每片号一份） | 共享后（1 份索引 + V 个 overlay） | 倍数 |
  |---|---|---|---|
  | 1 | 55,541,660 B | 55,541,724 B | 1.00× |
  | 2 | 111,083,320 B | 55,541,788 B | 2.00× |
  | 4 | 222,166,640 B | 55,541,916 B | 4.00× |
  | 8 | 444,333,280 B | 55,542,172 B | 8.00× |

  与 O(N + V + R log N) 一致：新增片号只 +~64 B（空 overlay），新增订单只 +~4 KB/单。
- **接/撤单耗时**（300 次接 + 300 次撤，中位数）：N=10,000 → **116.5 µs/次**；
  N=100,000 → **187.5 µs/次**。N 涨 10 倍耗时仅 ~1.6 倍，符合 O(log N)。
- **并发导入**（进程内）：4 个线程同时导入同一份 100,000 镜头内容，**只建 1 棵树**，
  2.75 s 全部返回，常驻 1 个 catalog / 53.0 MB；8 线程版本同样只建 1 棵树（测试断言）。
- **端到端（HTTP，本地 uvicorn）**：100,000 镜头导入首个片号 6.3 s（其中 ~1.6 s 为建树，其余为
  JSON 校验/反序列化），随后 3 个同内容片号各 4.8～5.2 s（纯校验，**零建树**）；
  4 个片号共 1 份 catalog：`retained_bytes_total` 55.5 MB vs `naive_bytes_total` 222.2 MB，
  节省 166.6 MB；100k 数据接/撤单经 HTTP 为 **1.7～2.0 ms**（含 JSON/HTTP 开销），
  另一片号查询 1.6 ms。首次在该片号 overlay 上落单时进程内耗时为 0.34 ms（节点分配），
  无单独的长尾。
- **演示**（`scripts/demo.sh`，本地 uvicorn 实跑）：3 个渠道片号同内容 → 1 个 catalog，`shared_videos` 1→2→3；
  `channel-a` 隐藏 [2,6)、`channel-b` 用同 `order_id` 隐藏 [0,4) 互不影响，`channel-c` 仍 10/10 可播；
  重送同范围幂等、改范围 409；全隐藏 → `withdrawn`；`channel-a/b/c` 依次迁走，
  前两个离开时旧 catalog 继续服务剩余片号，最后一个离开后 `catalog_count` 由 2 降为 1，旧索引释放。

## 未完成项 / 限制

1. **Docker 未实测**：本环境无 Docker 守护进程（`docker: command not found`），
   `Dockerfile` / `docker-compose.yml` 未实际构建运行（镜像为标准 `python:3.11-slim` +
   `requirements.txt`，compose 仅单服务 + healthcheck，风险低）。所有服务实测均用本地 uvicorn 完成。
2. **单进程内存态**：数据存于进程内存，重启即失；多 worker/多副本不共享状态
   （compose 固定单容器单 worker）。跨进程共享需外置存储或按内容指纹分片。
3. **内容比对方式**：按导入 payload（校验后的特征/分数）的 SHA-256 指纹去重，要求镜头顺序、
   数值完全一致；仅有浮点末位差异也会视为不同内容而各建一份索引。
4. **导入耗时**：100,000 镜头首次导入约 5～6 s（O(N) 建树 + JSON 校验/反序列化）；
   同内容的后续导入仍要做完整 JSON 校验，约 5 s，但不再建树、不再额外占索引空间。
   导入建树在锁外，不阻塞其他请求；CPU 本身是瓶颈。
5. **首次触达成本**：某片号的第一张单需要在其 overlay 里建立 O(log N) 路径节点，
   100k 数据下首次接单进程内约 0.34 ms；之后稳定在 0.05～0.2 ms（进程内）/ ~1.8 ms（含 HTTP）。
6. 无鉴权、无持久化、无速率限制；`order_id` 仅在单个片号内唯一。
