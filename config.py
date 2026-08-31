import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data" / "index"
DB_PATH = INDEX_DIR / "rag.sqlite3"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHAT_MODEL = "qwen3-4b-cuda-gpu"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 110
DENSE_CANDIDATES = 16
BM25_CANDIDATES = 16
RERANK_CANDIDATES = 8
RETRIEVAL_LIMIT = 4

MAX_CONTEXT_CHARS_PER_CHUNK = 750
MAX_OUTPUT_TOKENS = 420


def get_foundry_base_url() -> str:
    try:
        result = subprocess.run(
            ["foundry", "server", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        match = re.search(r"http://127\.0\.0\.1:\d+", f"{result.stdout}\n{result.stderr}")
        if match:
            return f"{match.group(0)}/v1"
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError(
        "Foundry Local sunucusu bulunamadı. `foundry server start` komutunu çalıştırın "
        "ve `foundry model load qwen3-4b-cuda-gpu:2` ile modeli yükleyin."
    )


FOUNDRY_BASE_URL = get_foundry_base_url()
