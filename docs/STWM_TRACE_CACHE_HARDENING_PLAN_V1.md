# STWM Trace Cache Hardening Plan V1

Date: 2026-04-03
Scope: Design-only, no runtime mutation of active jobs

## 1) 背景与目标

近期 D1 中两条 run 失败均发生在 trace cache 读取阶段（`BadZipFile` / `EOFError`），说明 trace cache 在并发写读场景下存在可靠性缺口。

本方案目标是“最小改动修复”而非重构：

1. 原子写入（tmp -> fsync -> rename）
2. 坏文件隔离（quarantine）
3. 读失败自动重建
4. 并发访问保护

## 2) 现状差异

当前 `trace_adapter.py`:

- 写缓存: 直接 `np.savez_compressed(cache_path, ...)`
- 读缓存: `np.load(cache_path)` 异常直接向上抛

对照 `semantic_adapter.py`:

- 已具备临时文件 + `os.replace` 原子替换
- 已具备坏缓存隔离（quarantine）
- 已具备可恢复错误识别后重建路径

## 3) 最小修复方案

### 3.1 原子写

把 trace cache 写路径改为：

1. 在同目录写临时文件 `*.tmp.<pid>.<ts>`
2. `flush()` + `os.fsync(fd)`
3. `os.replace(tmp, cache_path)`
4. 对父目录执行一次 `fsync(dir_fd)`（可选但推荐）

这样可避免读到半写入文件。

### 3.2 坏文件隔离（quarantine）

在读取失败时（`BadZipFile` / `EOFError` / `ValueError` / `OSError`），将坏文件移动到：

- `<cache_dir>/quarantine/<stem>.bad_<timestamp>_<digest>.npz`

并在 metadata/log 中记录隔离路径和异常类型。

### 3.3 读失败自动重建

读取流程改为：

1. 尝试读 cache
2. 若失败且属于可恢复异常:
   - quarantine 坏文件
   - 回退到源输入重算 summary
   - 走原子写回 cache
3. 返回重建结果

约束：仅对可恢复异常自动重建，其他异常仍抛出。

### 3.4 并发访问保护

引入每-key 锁文件（例如 `<cache_path>.lock`），使用 `fcntl.flock`:

- 写路径: `LOCK_EX`
- 读路径: 可先无锁快速读；若失败进入恢复路径时再 `LOCK_EX`

最小实现建议：恢复路径加互斥，避免多个进程同时重建同一 key。

## 4) 伪流程

```text
encode(...):
  if cache exists:
    try load
    except recoverable:
      acquire EX lock
      re-check cache
      try load again
      if still bad:
        quarantine
        rebuild_from_source
        atomic_save
      release lock
      return summary
  else:
    acquire EX lock
    re-check cache
    if exists and load ok: return
    rebuild_from_source
    atomic_save
    release lock
    return
```

## 5) 变更边界（最小改动）

仅建议改动：

1. `code/stwm/modules/trace_adapter.py`

不改动：

1. 训练主循环逻辑
2. 任务调度逻辑
3. 现有运行任务状态

## 6) 验收标准

1. 并发压测下不再出现 `BadZipFile` / `EOFError` 导致训练退出
2. 发生坏缓存时可自动隔离并重建，不中断训练
3. cache 命中路径性能不明显回退（命中场景额外开销可控）

## 7) 风险与回滚

风险:

1. 锁实现不当造成等待放大
2. 错误分类过宽导致误重建

回滚策略:

1. 通过 feature flag 关闭 trace cache hardening（保留旧路径）
2. 保留 quarantine 目录便于事后样本分析
