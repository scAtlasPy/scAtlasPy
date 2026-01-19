import os
from pathlib import Path
import urllib.request
import re
import subprocess
file_dir = Path(__file__).parent.resolve()
task_txt = open(file_dir / "tasks.txt", "w")

root_dir = Path(__file__).parent.resolve()
while not (root_dir / "config_vars.sh").is_file() and not root_dir.parent == root_dir:
    root_dir = root_dir.parent
config = {}
exec(open(root_dir / "config_vars.sh").read(), config)

# Perturb-seq is the slowest here, with about 2 hours to download
SBATCH_OPTS=" ".join([
    "-p wjg,biochem,sfgf,owners",
    "-c 2",
    "--time=4:00:00",
])
print(SBATCH_OPTS, file=task_txt)

SINGULARITY=config["CONTAINER_TEMPLATE"].format(CONTAINER="bpcells-v0.3.0", OMP_NUM_THREADS=1)
DATA_ROOT=config["DATA_ROOT"]
TMP_ROOT=config["TMP_ROOT"]

# Just download the raw input files into our raw download destinations
dataset_urls = {
    "20k_pbmc": ["https://cf.10xgenomics.com/samples/cell-exp/6.1.0/20k_PBMC_3p_HT_nextgem_Chromium_X/20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"],
    "130k_thymus_atlas": ["https://zenodo.org/record/5500511/files/HTA07.A01.v02.entire_data_raw_count.h5ad?download=1"],
    "480k_tabula_sapiens": ["https://figshare.com/ndownloader/files/34702114"],
    "500k_drugscreen": ["https://cf.10xgenomics.com/samples/cell-exp/6.1.0/H1975_A549_DrugScreen_3p_HT_nextgem/H1975_A549_DrugScreen_3p_HT_nextgem_count_filtered_feature_bc_matrix.h5"],
    "1m_neurons": ["https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5"],
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
    "22m_pansci": ["https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE247719&format=file&file=GSE247719%5F20240213%5FPanSci%5Fall%5Fcells%5Fadata%2Eh5ad%2Egz"]
}

for dataset, urls in dataset_urls.items():
    output_dir = f"{DATA_ROOT}/rna/downloads/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    command = [f"{SINGULARITY} curl --parallel --output-dir {output_dir} -LJ"]
    if dataset == "4m_fetal":
        # Unfortunately, the server configuration for the fetal atlas data website requires user-agent spoofing to return data
        # (Possibly a CloudFront configuration error?)
        command.append("--user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0'")
    for url in urls:
        command.append(f"-O \"{url}\"")
    print(" ".join(command), file=task_txt)