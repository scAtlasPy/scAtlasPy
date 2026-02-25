import duckdb
import threading
import time
import multiprocessing


# ==========================================================
# 每个线程执行的函数
# ==========================================================
def read_func_t(pid, tid, n_process, n_thread, file, fetch_rows):

    # 每个线程建立独立 DuckDB 连接
    conn = duckdb.connect(file, read_only=True)

    # ======================================================
    # SQL 分片逻辑：
    #
    # 1️⃣ id // fetch_rows         → 生成 block_id
    # 2️⃣ block_id % n_thread      → block 分给哪个线程
    # 3️⃣ (block_id // n_thread) % n_process   → block 分给哪个进程
    # 等价于：
    # 所有 block 均匀分布到 n_process × n_thread 个 worker
    # ======================================================
    results = conn.execute(f"""
        SELECT indices, data
        FROM (
            SELECT indices, data, id // {fetch_rows} AS block_id
            FROM X_CSR_data
        )
        WHERE (block_id // {n_thread}) % {n_process} = {pid}
          AND block_id % {n_thread} = {tid}
    """).fetch_record_batch(rows_per_batch=fetch_rows)

    # 计时开始
    t0 = time.time()

    # 统计读取了多少个 Arrow block
    n = 0
    for arrow in results:
        n += 1

    t1 = time.time()

    print(
        f"process {pid} thread {tid} "
        f"read {n} block, eta {n/(t1-t0):.2f} block/s"
    )


# ==========================================================
# 使用预计算索引字段的版本（理论上更快）
# ==========================================================
def read_func_use_index(pid, tid, n_process, n_thread, file, fetch_rows):

    conn = duckdb.connect(file, read_only=True)

    # 如果表中已经提前写入：
    #   proc_id
    #   thread_id
    #
    # 那就可以直接等值过滤
    # 这样避免了运行时计算 block_id
    results = conn.execute(f"""
        SELECT indices, data
        FROM X_CSR_data
        WHERE proc_id = {pid}
          AND thread_id = {tid}
    """).fetch_record_batch(rows_per_batch=fetch_rows)

    t0 = time.time()
    n = 0
    for arrow in results:
        n += 1
    t1 = time.time()

    print(
        f"process {pid} thread {tid} "
        f"read {n} block, eta {n/(t1-t0):.2f} block/s"
    )


# 预计算索引
def prepare_partition_index(file, n_process, n_thread, fetch_rows):
    """
    为 read_func_use_index 版本准备分区字段和索引

    1. 添加 proc_id, thread_id 列（如果不存在）
    2. 按 block 分片规则填充
    3. 创建复合索引 (proc_id, thread_id)
    """

    conn = duckdb.connect(file)

    print("Step 1: 添加列（如果不存在）...")

    # 添加列（如果不存在）
    conn.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS proc_id INTEGER;
    """)

    conn.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS thread_id INTEGER;
    """)

    print("Step 2: 填充分区字段...")

    # 按你原有的 block 分片规则填充
    conn.execute(f"""
        UPDATE X_CSR_data
        SET
            thread_id = (id // {fetch_rows}) % {n_thread},
            proc_id = ((id // {fetch_rows}) // {n_thread}) % {n_process};
    """)

    # print("Step 3: 创建复合索引...")
    #
    # conn.execute("""
    #     CREATE INDEX IF NOT EXISTS idx_proc_thread
    #     ON X_CSR_data (proc_id, thread_id);
    # """)

    conn.close()

    print("✅ 分区字段和索引准备完成")



def read_func_mod(tid,n_threads,file,fetch_size):

    conn = duckdb.connect(file, read_only=True)
    results = conn.execute(
        f"""
            SELECT indices, data
            FROM X_CSR_data
            WHERE (id // {fetch_size}) % {n_threads} = {tid};
        """
    ).fetch_record_batch(rows_per_batch=fetch_size)

    #
    # fetch_rows = 100 ； n_threads = 3
    # block_id = id // fetch_size  ——  整除：每个block有fetch_size 行
    # (block_id) % n_threads       ——   block_id 除以 n_threads 的余数
    # 0-99     → block 0 → 0 % 3 = 0 → thread 0
    # 100-199  → block 1 → 1 % 3 = 1 → thread 1
    # 200-299  → block 2 → 2 % 3 = 2 → thread 2
    # 300-399  → block 3 → 3 % 3 = 0 → thread 0
    # 400-499  → block 4 → 4 % 3 = 1 → thread 1
    # 结果：
    # thread 0 → block 0,3,6,9...
    # thread 1 → block 1,4,7...
    # thread 2 → block 2,5,8...

    t0=time.time()
    n=0
    for arrow in results:
        n+=1
    t1 = time.time()
    print(f"thread {tid} read {n} block, eta {n/(t1-t0)}")

# ==========================================================
# 每个进程执行的函数
# ==========================================================
def main_process(pid, n_process, n_thread, file, fetch_rows):

    # 每个进程创建 n_thread 个线程
    threads = [
        threading.Thread(
            target=read_func_use_index,
            kwargs=dict(
                pid=pid,
                tid=tid,
                n_process=n_process,
                n_thread=n_thread,
                file=file,
                fetch_rows=fetch_rows
            )
        )
        for tid in range(n_thread)
    ]

    # 启动所有线程
    for t in threads:
        t.start()

    # 等待所有线程结束
    for t in threads:
        t.join()


# ==========================================================
# 主程序
# ==========================================================
if __name__ == '__main__':

    n_process = 2
    n_thread = 5
    fetch_rows = 2048 * 2000

    file = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql"

    t0 = time.time()

    # prepare_partition_index_only_t(
    #     file,
    #     n_process=n_process,
    #     n_thread=n_thread,
    #     fetch_rows=fetch_rows
    # )

    # t0 = time.time()
    # 创建多个进程
    processes = [
        multiprocessing.Process(
            target=main_process,
            kwargs=dict(
                pid=pid,
                n_process=n_process,
                n_thread=n_thread,
                file=file,
                fetch_rows=fetch_rows
            )
        )
        for pid in range(n_process)
    ]

    t0 = time.time()

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    t1 = time.time()
    print(f"overall time: {t1-t0:.3f}s")


# 索引
# 进程 + 线程   2 * 5  1.199s
# 只用多线程    10线程  1.364s