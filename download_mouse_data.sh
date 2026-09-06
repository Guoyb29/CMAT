#!/usr/bin/env bash
# Download mouse embryo development data for CMAT experiments

set -e  # Exit on error

DATA_DIR="./data/mouse"
mkdir -p "$DATA_DIR"

echo "========================================="
echo "Downloading mouse embryo data for CMAT"
echo "========================================="

# E7.5 trimodal data (Argelaguet et al., 2019)
echo ""
echo "[1/3] Downloading E7.5 trimodal scNMT-seq data..."
if [ ! -f "$DATA_DIR/E75_scNMT.h5ad" ]; then
    wget -O "$DATA_DIR/E75_scNMT.h5ad" \
        "https://content.cruk.cam.ac.uk/jmlab/mousetimedata/e75_trimodal.h5ad" \
        || echo "? Warning: E7.5 download failed. Please download manually."
else
    echo "  ? E7.5 data already exists, skipping."
fi

# E9.5 validation data (Cao et al., 2019)
echo ""
echo "[2/3] Downloading E9.5 validation data..."
if [ ! -f "$DATA_DIR/E95_real_converted.h5ad" ]; then
    echo "  ? E9.5 data requires manual download from:"
    echo "    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119945"
    echo "  Please download GSE119945_gene_count.txt.gz and convert to h5ad format."
else
    echo "  ? E9.5 data already exists, skipping."
fi

# E12.5 spatial reference (Chen et al., 2022)
echo ""
echo "[3/3] Downloading E12.5 spatial transcriptomics data..."
if [ ! -f "$DATA_DIR/E125_spatial.h5ad" ]; then
    echo "  ? E12.5 spatial data requires manual download from:"
    echo "    https://db.cngb.org/stomics/datasets/STDS0000058"
    echo "  Please download and convert to h5ad format."
else
    echo "  ? E12.5 data already exists, skipping."
fi

echo ""
echo "========================================="
echo "? Data download completed"
echo "========================================="
echo "Data location: $DATA_DIR"
echo ""
echo "Note: Some datasets require manual download and format conversion."
echo "Please refer to README.md for detailed instructions."
