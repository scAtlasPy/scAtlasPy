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
# 版本3：直接 s  e
# ==================================================
def read_func_s_e(tid, n_threads, file, fetch_size, total_rows):

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

    n_threads = 5
    file = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql"
    fetch_size = 2048 * 2000

    # 读取总行数
    conn = duckdb.connect(file, read_only=True)
    total_rows = conn.execute("SELECT COUNT(*) FROM X_CSR_data").fetchone()[0]
    conn.close()

    print(f"Total rows: {total_rows}")
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