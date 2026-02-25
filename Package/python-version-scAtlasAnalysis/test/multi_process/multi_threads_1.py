import duckdb
import threading
from tqdm import tqdm
import time

# ==================================================
# 版本1：表达式分片 (原始版本)
# ==================================================
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


# 修1
# ==================================================
# 版本1：MOD 等价的滑动窗口版本
# ==================================================
def read_func_mod_1(tid, n_threads, file, fetch_size, total_rows):

    conn = duckdb.connect(file, read_only=True)

    # block 总数
    total_blocks = (total_rows + fetch_size - 1) // fetch_size

    t0 = time.time()
    n = 0

    # 从 tid 开始，每次跳 n_threads 个 block
    for block_id in range(tid, total_blocks, n_threads):

        start = block_id * fetch_size
        end = min(start + fetch_size - 1, total_rows - 1)

        results = conn.execute(
            f"""
                SELECT indices, data
                FROM X_CSR_data
                WHERE id BETWEEN {start} AND {end};
            """
        ).fetch_record_batch(rows_per_batch=fetch_size)

        for arrow in results:
            n += 1

    t1 = time.time()
    print(f"[MOD-SLIDE] thread {tid} read {n} block, eta {n/(t1-t0):.2f} block/s")

# ==================================================
# 版本2：MOD 等价滑动窗口（无 fetch_record_batch）
# ==================================================
def read_func_mod_2(tid, n_threads, file, fetch_size, total_rows):

    conn = duckdb.connect(file, read_only=True)

    total_blocks = (total_rows + fetch_size - 1) // fetch_size

    t0 = time.time()
    n = 0

    for block_id in range(tid, total_blocks, n_threads):

        start = block_id * fetch_size
        end = min(start + fetch_size - 1, total_rows - 1)

        table = conn.execute(
            f"""
                SELECT indices, data
                FROM X_CSR_data
                WHERE id BETWEEN {start} AND {end};
            """
        ).fetch_arrow_table()
        #  todo 用这个来改进 单进程 版本？

        if table.num_rows > 0:
            n += 1

    t1 = time.time()
    print(f"[MOD-SLIDE] thread {tid} read {n} block, eta {n/(t1-t0):.2f} block/s")


# todo 只用多线程
def read_func_use_index_only_t( tid, n_threads, file, fetch_size):

    conn = duckdb.connect(file, read_only=True)

    results = conn.execute(f"""
        SELECT indices, data
        FROM X_CSR_data
        WHERE thread_id_only1 = {tid}
    """).fetch_record_batch(rows_per_batch=fetch_size)

    t0 = time.time()
    n = 0
    for arrow in results:
        n += 1
    t1 = time.time()

    print(
        f"thread {tid} "
        f"read {n} block, eta {n/(t1-t0):.2f} block/s"
    )

def prepare_partition_index_only_t(file, n_thread, fetch_rows):
    """
    为 read_func_use_index 版本准备分区字段和索引

    1. 添加 thread_id 列（如果不存在）
    2. 按 block 分片规则填充
    """

    conn = duckdb.connect(file)

    print("Step 1: 添加列（如果不存在）...")

    # 添加列（如果不存在）
    conn.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS thread_id_only1 INTEGER;
    """)

    print("Step 2: 填充分区字段...")

    # 按你原有的 block 分片规则填充
    conn.execute(f"""
        UPDATE X_CSR_data
        SET
            thread_id_only1 = (id // {fetch_rows}) % {n_thread}
    """)

    conn.close()

    print("✅ 分区字段和索引准备完成")

# todo 只用多线程

# ==================================================
# 版本2：Range 分片（连续区间）
# ==================================================
def read_func_range(tid, n_threads, file, fetch_size, total_rows):

    conn = duckdb.connect(file, read_only=True)

    # 每个线程负责一个连续区间
    chunk = total_rows // n_threads
    start = tid * chunk
    end = total_rows - 1 if tid == n_threads - 1 else (start + chunk - 1)

    results = conn.execute(
        f"""
            SELECT indices, data
            FROM X_CSR_data
            WHERE id BETWEEN {start} AND {end};
        """
    ).fetch_record_batch(rows_per_batch=fetch_size)

    t0 = time.time()
    n = 0
    for arrow in results:
        n += 1
    t1 = time.time()

    print(f"[RANGE] thread {tid} read {n} blocks, eta {n/(t1-t0):.2f} block/s")

# ==================================================
# 主函数
# ==================================================
if __name__ == '__main__':

    n_threads = 10
    file = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql"
    fetch_size = 2048 * 2000

    # 读取总行数
    conn = duckdb.connect(file, read_only=True)
    total_rows = conn.execute("SELECT COUNT(*) FROM X_CSR_data").fetchone()[0]
    conn.close()

    print(f"Total rows: {total_rows}")
    print("=" * 60)

    # # todo 建立多线程索引
    # prepare_partition_index_only_t(file, n_threads, fetch_size)

    # ===============================
    # todo 测试 线程索引 分片
    # ===============================
    threads = [
        threading.Thread(
            target=read_func_use_index_only_t,
            kwargs=dict(
                tid=i,
                n_threads=n_threads,
                file=file,
                fetch_size=fetch_size
            )
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    print(f"[index ] overall time: {t1 - t0:.3f}s")
    print("=" * 60)

    # ===============================
    # 1️⃣ 测试 MOD 分片
    # ===============================
    threads = [
        threading.Thread(
            target=read_func_mod,
            kwargs=dict(
                tid=i,
                n_threads=n_threads,
                file=file,
                fetch_size=fetch_size
            )
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    print(f"[MOD ] overall time: {t1 - t0:.3f}s")
    print("=" * 60)

    # ===============================
    # 2️⃣ 测试 RANGE 分片
    # ===============================
    threads = [
        threading.Thread(
            target=read_func_mod_1,
            kwargs=dict(
                tid=i,
                n_threads=n_threads,
                file=file,
                fetch_size=fetch_size,
                total_rows=total_rows
            )
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    print(f"[MOD 1] overall time: {t1 - t0:.3f}s")

    # ===============================
    # 2️⃣ 测试 RANGE 分片
    # ===============================
    threads = [
        threading.Thread(
            target=read_func_mod_2,
            kwargs=dict(
                tid=i,
                n_threads=n_threads,
                file=file,
                fetch_size=fetch_size,
                total_rows=total_rows
            )
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    print(f"[MOD 2]overall time: {t1 - t0:.3f}s")

    # ===============================
    # 2️⃣ 测试 RANGE 分片
    # ===============================
    threads = [
        threading.Thread(
            target=read_func_range,
            kwargs=dict(
                tid=i,
                n_threads=n_threads,
                file=file,
                fetch_size=fetch_size,
                total_rows=total_rows
            )
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    print(f"[RANGE] overall time: {t1 - t0:.3f}s")

    # MOD 版本 ≈ n_threads 倍 IO
    # RANGE 版本 ≈ 1 倍 IO
    # MOD 分片- xh	  RANGE 分片-yyz
    # 可能全表扫描	 只扫描自己区间
    # 表达式计算	     直接范围过滤
    # IO 放大	      IO 减少