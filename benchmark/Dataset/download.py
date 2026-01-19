import os
from pathlib import Path
import requests
import re
from urllib.parse import urlparse

# 定义脚本所在目录和下载目标目录
SCRIPT_DIR = Path(__file__).parent.resolve()
DOWNLOAD_ROOT = SCRIPT_DIR
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)

# 针对某些服务器需要添加 User-Agent
USER_AGENT = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0'

# 恢复所有数据集的下载URL列表
dataset_urls = {
    "20k_pbmc": [
        "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/20k_PBMC_3p_HT_nextgem_Chromium_X/20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"],
    "130k_thymus_atlas": [
        "https://zenodo.org/record/5500511/files/HTA07.A01.v02.entire_data_raw_count.h5ad?download=1"],
    "480k_tabula_sapiens": ["https://figshare.com/ndownloader/files/34702114"],
    "500k_drugscreen": [
        "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/H1975_A549_DrugScreen_3p_HT_nextgem/H1975_A549_DrugScreen_3p_HT_nextgem_count_filtered_feature_bc_matrix.h5"],
    "1m_neurons": [
        "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5"],
    "2m_perturbseq": ["https://plus.figshare.com/ndownloader/files/35775507"],
    "4m_fetal": [
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/df_cell.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Adrenal_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Cerebellum_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Cerebrum_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Eye_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Heart_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Intestine_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Kidney_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Liver_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Lung_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Muscle_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Pancreas_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Placenta_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/293T_3T3_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Spleen_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Stomach_gene_count.RDS",
        "https://atlas.fredhutch.org/data/bbi/descartes/human_gtex/downloads/data_summarize_fetus_data/Thymus_gene_count.RDS",
    ],
    "11m_jax": [
        "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/adata/adata_JAX_dataset_1.h5ad",
        "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/adata/adata_JAX_dataset_2.h5ad",
        "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/adata/adata_JAX_dataset_3.h5ad",
        "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/adata/adata_JAX_dataset_4.h5ad",
    ],
    "22m_pansci": [
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE247719&format=file&file=GSE247719%5F20240213%5FPanSci%5Fall%5Fcells%5Fadata%2Eh5ad%2Egz"]
}


def get_filename_from_response(r, dataset_name, url):
    """尝试从 Content-Disposition 或 URL 中提取文件名"""
    filename = None

    # 1. 从 Content-Disposition 获取文件名（最准确）
    if 'content-disposition' in r.headers:
        cd = r.headers['content-disposition']
        # 匹配 filename="example.ext"
        fname_match = re.search(r'filename="(.+?)"', cd)
        if fname_match:
            filename = fname_match.group(1)

    # 2. 如果没有 Content-Disposition，从最终 URL 路径获取
    if not filename:
        # 使用 r.url (重定向后的最终 URL)
        filename = Path(urlparse(r.url).path).name

    # 3. 如果文件名仍然不明确，使用 URL 的最后一部分作为回退
    if not filename:
        filename = url.split('/')[-1].split('?')[0]

    # 清理文件名中的非法字符 (简单处理)
    filename = re.sub(r'[\s/\\:*?"<>|]', '_', filename).strip('_')

    # 确保文件名不为空，并添加数据集前缀以防冲突
    if not filename:
        filename = "unknown_file"

    return f"{dataset_name}_{filename}"


def download_file(url, output_dir, dataset_name):
    """使用 requests 库下载文件"""

    headers = {}
    if dataset_name == "4m_fetal":
        headers['User-Agent'] = USER_AGENT

    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            # 获取处理后的文件名
            final_filename = get_filename_from_response(r, dataset_name, url)
            output_path = output_dir / final_filename

            print(f"    下载 URL: {r.url}")
            print(f"    保存至: {output_path.name}")

            # 检查文件是否已存在 (跳过已下载的文件)
            if output_path.exists():
                # 简单检查大小，如果大于 0 字节，则跳过
                if output_path.stat().st_size > 0:
                    print(f"    [跳过] 文件已存在且非空：{output_path.name}")
                    return

            # 写入文件
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"    [成功] {output_path.name}")

    except requests.exceptions.RequestException as e:
        print(f"    [失败] 下载 {url} 失败: {e}")


print(f"==========================================")
print(f"开始下载所有数据集到目录: {DOWNLOAD_ROOT}")
print(f"==========================================")

for dataset, urls in dataset_urls.items():
    print(f"\n--- 处理数据集: {dataset} (共 {len(urls)} 个文件) ---")

    for url in urls:
        download_file(url, DOWNLOAD_ROOT, dataset)

print("\n所有下载任务完成。")
