#!/usr/bin/env bash
set -euo pipefail

# ADL2025 HW3 - download only (no installs, no tricks)
# Output structure required by TAs:
# models/
#   ├─ retriever/  (your fine-tuned bi-encoder folder)
#   └─ reranker/   (your fine-tuned cross-encoder folder)

mkdir -p models

echo "[download.sh] downloading retriever.zip ..."
gdown --id 11skjZoDbK26qy8mJwAOPqJb9b3nYSbty -O retriever.zip

echo "[download.sh] downloading reranker.zip ..."
gdown --id 1MMvMRpD0P4WME-sPHv5HeuHWGMGCQkpI -O reranker.zip

# zip 內容建議長這樣：
# retriever.zip  -> retriever/...(整個資料夾)
# reranker.zip   -> reranker/...(整個資料夾)

unzip -q retriever.zip -d models
unzip -q reranker.zip -d models
rm -f retriever.zip reranker.zip

# 輕量驗證（非必要，但不會違規：只是列檔案）
echo "[download.sh] contents under models/:"
ls -lah models || true

echo "[download.sh] ✅ done."
