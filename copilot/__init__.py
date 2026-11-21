import os

# faiss-cpu and torch each bundle libomp on macOS; without this the second load aborts the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
