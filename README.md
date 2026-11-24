# 片库镜头检索向量服务

用可播镜头实时维护视频检索向量：导入按时间排序的 1～100,000 个镜头（每个带 16 维特征、注意力分数），
按单号登记/撤销镜头隐藏区间，每次变更后以 **O(log N)** 返回由可播镜头算出的检索向量
（注意力分数 softmax 加权特征和）。

## 运行

```bash
# Docker Compose（推荐）
docker compose up --build

# 或本地运行
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务监听 `:8000`，健康检查 `GET /health`，交互文档 `GET /docs`。

演示脚本（重叠区间的登记与撤销，含幂等重送、冲突、全隐藏撤下、恢复）：

```bash
bash scripts/demo.sh            # BASE=http://localhost:8000 可覆盖
```

自测：

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/videos/{video_id}/shots` | 导入镜头。body: `{"shots":[{"feature":[16 个 −1~1],"score":−1000~1000}, ...]}`，1～100,000 条。重复导入整体替换并清空该视频所有订单 |
| `POST` | `/videos/{video_id}/restrictions` | 登记隐藏单。body: `{"order_id":"lic-A","start":2,"end":6}`，区间 0 起算、左闭右开 |
| `DELETE` | `/videos/{video_id}/restrictions/{order_id}` | 撤销一单；仍被其他单覆盖的镜头保持隐藏 |
| `GET` | `/videos/{video_id}/retrieval-vector` | 查询当前检索向量 |
| `GET` | `/health` | 健康检查 |

统一响应：

```json
{"video_id":"demo","shot_count":10,"visible_count":6,"status":"ok","vector":[...16 维...]}
```

- **幂等**：同 `order_id` 同范围重送 → 200，状态不变。
- **冲突**：同 `order_id` 不同范围 → 409（提示先撤销）。
- **全隐藏**：`status="withdrawn"`、`vector=null` —— 即撤下索引的标记；撤销后自动恢复。
- 校验失败 422（维度数、特征/分数越界、镜头数越界），区间语义错误 400（`start>=end`、`end>N`），未知视频/单号 404。

## 实现（`app/segment_tree.py`）

线段树维护可播集合上的稳定 softmax 加权和。每个节点存：

- `cnt`：可播镜头数
- `mx`：可播注意力分数最大值
- `sm = Σ exp(score − mx)`
- `vec[16] = Σ exp(score − mx) · feature`

兄弟合并时平移到共同最大值（经典 online-softmax 分解）：

```
m   = max(mx_l, mx_r)
sm  = sm_l·exp(mx_l−m) + sm_r·exp(mx_r−m)
vec = vec_l·exp(mx_l−m) + vec_r·exp(mx_r−m)
```

隐藏用**懒引用计数**：登记对区间覆盖的 O(log N) 个规范节点 `cover += 1`，撤销 `cover -= 1`；
`cover > 0` 的节点贡献为空。多单重叠时同一镜头计数 ≥1，撤销一单后若仍被其他单覆盖则保持隐藏。
子节点聚合始终有效，无需下推。检索向量 = 根 `vec / sm`（逐维）。

**复杂度**：登记/撤销各 O(log N) 次节点访问（每次合并为 16 维常数工作）；导入建树 O(N)；查询 O(1)。
N=100,000 时树约 2¹⁸ 个节点，扁平 `array('d')` 存储，内存约 40 MB/视频。

**精度**：合并只做 max 平移与浮点乘加，与"稳定全量计算"（max 平移 + `math.fsum` 精确求和的参考实现）
的差异仅来自求和结合顺序，误差随树高 ~log₂N 线性累积，理论量级 ~1e-15，远小于 1e-9 的验收阈值。
分数区间 [−1000, 1000] 内 `exp` 不会上溢，下溢到 0 与全量计算行为一致。

## 实测结果（本机 Python 3.11，pytest 9.2）

- **47 个测试全部通过**（`tests/test_service.py`）：
  - 40 组随机用例（N=1…1000，每步登记/撤销后比对参考实现）：逐维误差 ≤ 1e-9 ✅
  - 极端值（分数 ±1000、特征 ±1）✅；N=100,000 大样本 ✅；全隐藏→恢复 ✅
  - API：重叠撤销、幂等重送、409 冲突、404、422/400 校验、withdrawn 标记 ✅
- **精度实测**：N=100,000 下 60 个随机隐藏状态，逐维最大误差 **8.3e-17**（阈值 1e-9）。
- **复杂度实测**（2000 次登记 + 2000 次撤销取平均）：
  - N=10,000 → 89.7 µs/次；N=100,000 → 181.5 µs/次。N 涨 10 倍耗时仅 ~2 倍，符合 O(log N)。
- **端到端（HTTP）**：导入 100,000 镜头（16 MB body）3.7 s；登记/撤销单次 1.5～10 ms；
  撤销重叠单后 `visible_count` 正确；全隐藏返回 `{"status":"withdrawn","vector":null}`。
- **演示**：`scripts/demo.sh` 输出见上节描述，撤销 lic-A 后 [4,8) 仍被 lic-B 隐藏，
  全部撤销后向量与导入时逐位一致。

## 未完成项 / 限制

1. **Docker 未实测**：本环境无 Docker 守护进程，`Dockerfile` / `docker-compose.yml` 未实际构建运行
  （已做 YAML 语法校验；镜像为标准 `python:3.11-slim` + `requirements.txt`，风险低）。
2. **单进程内存态**：数据存于进程内存，重启即失；多 worker/多副本不共享状态（compose 固定单容器单 worker）。
   生产化需外置存储或分片路由。
3. **导入耗时**：100,000 镜头导入约 2～4 s（O(N) 建树 + JSON 校验），导入期间会阻塞其他请求
  （FastAPI 线程池内持锁仅覆盖换指针，瓶颈在 CPU 本身）。
4. 重复导入同一 `video_id` 会**清空其全部订单**（按"整体替换"语义实现，如需保留可改）。
5. 无鉴权、无持久化、无速率限制；`order_id` 仅在单个视频内唯一。
