"""
RAG retriever — fetches relevant ROCm docs from ChromaDB.

Used to inject ROCm API references into translation prompts. Dramatically
improves translation accuracy because the model doesn't need to memorize
ROCm API details — it reads them from context.

Auto-initialization: If ChromaDB is empty, the retriever auto-populates it
from the curated chunks in scripts/build_rag.py on first use. This means
the RAG works out-of-the-box without requiring the team to run build_rag.py
manually before the hackathon.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.config import settings


class RAGRetriever:
    """ChromaDB-backed RAG retriever with auto-initialization."""

    # Maximum retry attempts on ChromaDB connection failure
    MAX_INIT_RETRIES = 3

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False
        self._init_failed = False  # Track permanent failure

    def _initialize(self):
        """Initialize ChromaDB client (lazy, with retry)."""
        if self._initialized:
            return

        if self._init_failed:
            # Don't keep retrying after a permanent failure
            return

        for attempt in range(self.MAX_INIT_RETRIES):
            try:
                import chromadb

                # Embedding function — try sentence-transformers first, fall back to default
                self._embedding_fn = None
                try:
                    from chromadb.utils import embedding_functions
                    self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=settings.embedding_model,
                    )
                except (ImportError, Exception) as e:
                    logger.info(f"sentence-transformers not available ({e}), using ChromaDB default embedding")
                    self._embedding_fn = None  # ChromaDB will use its default (all-MiniLM-L6-v2)

                # Client
                if settings.chroma_available:
                    # Remote ChromaDB (docker-compose service)
                    self._client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                    )
                    logger.info(f"ChromaDB: connected to {settings.chroma_host}:{settings.chroma_port}")
                else:
                    # Local persistent
                    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=settings.chroma_persist_dir,
                    )
                    logger.info(f"ChromaDB: local persistent at {settings.chroma_persist_dir}")

                # Get or create collection (use embedding_fn if available, else default)
                if self._embedding_fn is not None:
                    self._collection = self._client.get_or_create_collection(
                        name="rocm_docs",
                        embedding_function=self._embedding_fn,
                        metadata={"description": "ROCm 7.2.3 documentation and API references"},
                    )
                else:
                    self._collection = self._client.get_or_create_collection(
                        name="rocm_docs",
                        metadata={"description": "ROCm 7.2.3 documentation and API references"},
                    )

                self._initialized = True

                # Auto-populate if empty
                if self._collection.count() == 0:
                    logger.info("RAG collection empty — auto-populating from curated chunks...")
                    self._auto_populate()

                logger.info(f"RAG initialized: {self._collection.count()} chunks in collection")
                return

            except Exception as e:
                logger.warning(f"RAG init attempt {attempt + 1}/{self.MAX_INIT_RETRIES} failed: {e}")
                if attempt == self.MAX_INIT_RETRIES - 1:
                    logger.warning(
                        f"RAG initialization failed after {self.MAX_INIT_RETRIES} attempts. "
                        f"RAG will return empty context. Translation quality may be reduced."
                    )
                    self._init_failed = True
                    return

    def _auto_populate(self):
        """Auto-populate the RAG collection from curated chunks.

        Imports the CURATED_CHUNKS list from scripts/build_rag.py and adds them
        to the collection. This means RAG works out-of-the-box without requiring
        the team to run build_rag.py manually.
        """
        try:
            # Add scripts/ to path so we can import build_rag
            import sys
            scripts_dir = Path(__file__).parent.parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from build_rag import CURATED_CHUNKS  # type: ignore

            documents = [chunk["text"] for chunk in CURATED_CHUNKS]
            metadatas = [chunk.get("metadata", {}) for chunk in CURATED_CHUNKS]

            # Use stable IDs based on chunk index (idempotent — safe to re-run)
            ids = [f"curated_{i}" for i in range(len(documents))]

            # Use upsert (idempotent — works whether or not chunks already exist)
            self._collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Auto-populated RAG with {len(documents)} curated chunks")

        except ImportError as e:
            logger.warning(f"Could not import CURATED_CHUNKS from build_rag.py: {e}")
        except Exception as e:
            logger.warning(f"Auto-populate failed: {e}")

    def retrieve(
        self,
        cuda_source: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Retrieve relevant ROCm docs for a CUDA source.

        Args:
            cuda_source: CUDA source code (used as the query).
            top_k: Number of chunks to retrieve. Defaults to settings.rag_top_k.

        Returns:
            List of relevant text chunks. Empty list if RAG is unavailable.
        """
        self._initialize()

        if self._collection is None:
            return []

        try:
            # Build a query from the CUDA source
            query = self._build_query(cuda_source)

            results = self._collection.query(
                query_texts=[query],
                n_results=top_k or settings.rag_top_k,
            )

            chunks = results.get("documents", [[]])[0]
            logger.debug(f"RAG retrieved {len(chunks)} chunks for query: {query[:80]}...")
            return chunks

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return []

    def _build_query(self, cuda_source: str) -> str:
        """Build a search query from CUDA source.

        Extracts:
        - Library calls (cublas, cudnn, thrust) — these map to ROCm docs
        - CUDA API calls (cudaMalloc, etc.)
        - Include headers (which libraries are used)
        - First 200 chars of source for context
        """
        import re

        parts = []

        # Library calls
        lib_calls = re.findall(r"\b(cublas\w+|cudnn\w+|thrust::\w+)\b", cuda_source)
        if lib_calls:
            parts.append(" ".join(set(lib_calls)))

        # CUDA API calls
        cuda_calls = re.findall(r"\bcuda(\w+)\s*\(", cuda_source)
        if cuda_calls:
            parts.append(" ".join(set(f"cuda{c}" for c in cuda_calls)))

        # Include headers
        includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', cuda_source)
        if includes:
            parts.append(" ".join(includes))

        # Shared memory / warp primitives
        if "__shared__" in cuda_source:
            parts.append("shared memory")
        if "__shfl" in cuda_source:
            parts.append("warp shuffle primitives")

        # Source preview
        parts.append(cuda_source[:200])

        return " ".join(parts)[:1000]  # Cap at 1000 chars

    def add_documents(self, documents: List[str], metadatas: List[dict] | None = None):
        """Add documents to the ChromaDB collection.

        Used by scripts/build_rag.py to populate the RAG corpus.
        Uses upsert (idempotent — safe to re-run).
        """
        self._initialize()

        if self._collection is None:
            logger.error("Cannot add documents: RAG not initialized")
            return

        # Generate stable IDs
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Use upsert for idempotency
        self._collection.upsert(
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids,
        )
        logger.info(f"Added {len(documents)} documents to RAG collection")

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        self._initialize()
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
