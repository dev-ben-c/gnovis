"""SQLite + FTS5 + vector memory store with hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("engram")

# Duplicate detection: L2 distance threshold for same-model similarity check.
# For normalized 768-dim embeddings, L2 < 0.4 ≈ cosine similarity > 0.92.
DUPLICATE_DISTANCE_THRESHOLD = 0.4

# ACT-R decay parameter (d ≈ 0.5 is standard in the literature).
ACTR_DECAY = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(content: str, salt: str = "") -> str:
    return hashlib.sha256(f"{content}{salt}{time.time_ns()}".encode()).hexdigest()[:12]


def model_family(model: str | None) -> str | None:
    """Extract model family: 'claude-opus-4-6' -> 'claude', 'qwen3:32b' -> 'qwen3'."""
    if not model:
        return None
    return model.split(":")[0].split("-")[0]


@dataclass
class Memory:
    id: str
    content: str
    memory_type: str  # "fact", "episode", "preference", "diary"
    category: str
    key: Optional[str]
    tags: list[str]
    confidence: float
    source: Optional[str]
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    model: Optional[str] = None  # which model created/last updated this
    context: Optional[str] = None  # reasoning/justification for the memory
    host: Optional[str] = None  # hostname this memory applies to (NULL = host-agnostic)
    score: float = 0.0  # populated during recall

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoryEntry:
    id: str
    memory_id: str
    action: str  # "created", "updated", "forgotten"
    model: Optional[str]
    old_content: Optional[str]
    new_content: Optional[str]
    old_confidence: Optional[float]
    new_confidence: Optional[float]
    context: Optional[str]  # reasoning for the change
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relationship:
    id: str
    entity_from: str
    entity_to: str
    relation_type: str
    metadata: Optional[dict]
    confidence: float
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Challenge:
    id: str
    target_memory_id: str
    challenger_model: str
    status: str  # "open", "accepted", "rejected", "synthesized"
    resolution: Optional[str]
    resolved_by: Optional[str]
    created_at: str
    resolved_at: Optional[str]
    arguments: list["ChallengeArgument"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arguments"] = [a.to_dict() for a in self.arguments]
        return d


@dataclass
class ChallengeArgument:
    id: str
    challenge_id: str
    model: str
    position: str  # "challenge", "defense", "rebuttal", "synthesis"
    argument: str
    evidence: list[str]  # memory IDs cited
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class DuplicateMemoryError(Exception):
    """Raised when content is semantically similar to an existing same-model memory."""

    def __init__(self, existing: "Memory", distance: float):
        self.existing = existing
        self.distance = distance
        # Approximate cosine similarity from L2 distance (assumes normalized vectors)
        self.similarity = max(0.0, 1.0 - distance**2 / 2.0)
        super().__init__(
            f"Similar memory already exists [{existing.id}] "
            f"(~{self.similarity:.0%} similar). Use 'update' to modify it."
        )


@dataclass
class RecallResult:
    """Partitioned recall results for provenance-aware retrieval."""
    own: list[Memory]       # caller's model family
    others: list[Memory]    # different model family
    unknown: list[Memory]   # no model recorded

    @property
    def all(self) -> list[Memory]:
        combined = self.own + self.others + self.unknown
        combined.sort(key=lambda m: m.score, reverse=True)
        return combined

    @property
    def total(self) -> int:
        return len(self.own) + len(self.others) + len(self.unknown)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK(memory_type IN ('fact', 'episode', 'preference', 'diary')),
    category TEXT NOT NULL DEFAULT 'general',
    key TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT 'legacy',
    context TEXT,
    host TEXT,
    UNIQUE(category, key, model)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, category, key, tags,
    content=memories, content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, category, key, tags)
    VALUES (new.rowid, new.content, new.category, new.key, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
    VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
    INSERT INTO memories_fts(rowid, content, category, key, tags)
    VALUES (new.rowid, new.content, new.category, new.key, new.tags);
END;

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    entity_from TEXT NOT NULL,
    entity_to TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    metadata TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(entity_from, entity_to, relation_type)
);

CREATE TABLE IF NOT EXISTS memory_history (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('created', 'updated', 'forgotten')),
    model TEXT,
    old_content TEXT,
    new_content TEXT,
    old_confidence REAL,
    new_confidence REAL,
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_memory ON memory_history(memory_id);
CREATE INDEX IF NOT EXISTS idx_history_model ON memory_history(model);
CREATE INDEX IF NOT EXISTS idx_history_created ON memory_history(created_at);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(category, key);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_model ON memories(model);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(entity_from);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(entity_to);

CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    target_memory_id TEXT NOT NULL,
    challenger_model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open', 'accepted', 'rejected', 'synthesized')),
    resolution TEXT,
    resolved_by TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS challenge_arguments (
    id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    position TEXT NOT NULL
        CHECK(position IN ('challenge', 'defense', 'rebuttal', 'synthesis')),
    argument TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_challenges_target ON challenges(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_challenges_status ON challenges(status);
CREATE INDEX IF NOT EXISTS idx_challenge_args_challenge ON challenge_arguments(challenge_id);
"""

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
OLLAMA_URL = "http://localhost:11434/api/embed"


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None, ollama_url: str | None = None):
        if db_path is None:
            db_path = Path.home() / ".engram" / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ollama_url = ollama_url or OLLAMA_URL
        self._embed_client = None  # lazy httpx.Client
        self._vec_available = self._load_sqlite_vec()
        self._init_schema()

    def _load_sqlite_vec(self) -> bool:
        """Load sqlite-vec extension. Returns True if available."""
        try:
            import sqlite_vec
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except (ImportError, Exception) as e:
            logger.warning(f"sqlite-vec not available, vector search disabled: {e}")
            return False

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._migrate()
        if self._vec_available:
            self._init_vec_table()
        self._conn.commit()

    def _init_vec_table(self):
        """Create the vec0 virtual table for KNN search if it doesn't exist."""
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'vec_memories'"
        ).fetchone()
        if not existing:
            try:
                self._conn.execute(f"""
                    CREATE VIRTUAL TABLE vec_memories USING vec0(
                        memory_id TEXT PRIMARY KEY,
                        embedding float[{EMBED_DIM}]
                    )
                """)
            except sqlite3.OperationalError as e:
                # Database locked by another process — will retry on next startup
                logger.warning(f"Could not create vec_memories table (will retry on next startup): {e}")
                self._vec_available = False

    def _schema_version(self) -> int:
        """Get current schema version (0 if no schema_version table)."""
        try:
            row = self._conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row and row[0] else 0
        except sqlite3.OperationalError:
            return 0

    def _migrate(self):
        """Run schema migrations up to the latest version."""
        # v0 → v1: add model and context columns (legacy migration)
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        for col, col_type in [("model", "TEXT"), ("context", "TEXT")]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_type}")

        version = self._schema_version()

        if version < 2:
            self._migrate_v2()

        if version < 3:
            self._migrate_v3()

        if version < 4:
            self._migrate_v4()

        if version < 5:
            self._migrate_v5()

    def _migrate_v2(self):
        """v2: Per-model memory — UNIQUE(category, key, model), model NOT NULL."""
        logger.info("Running schema migration v2: per-model memory partitioning")

        # Disable FK checks during migration (table rebuild breaks FK references)
        self._conn.execute("PRAGMA foreign_keys=OFF")

        # 1. Create schema_version table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            )
        """)

        # 2. Set model='legacy' on any NULL-model memories
        self._conn.execute("UPDATE memories SET model = 'legacy' WHERE model IS NULL")

        # 3. Rebuild memories table with new constraints
        #    SQLite requires table rebuild for constraint changes
        self._conn.execute("DROP TRIGGER IF EXISTS memories_ai")
        self._conn.execute("DROP TRIGGER IF EXISTS memories_ad")
        self._conn.execute("DROP TRIGGER IF EXISTS memories_au")

        self._conn.execute("ALTER TABLE memories RENAME TO memories_old")

        self._conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('fact', 'episode', 'preference')),
                category TEXT NOT NULL DEFAULT 'general',
                key TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT 'legacy',
                context TEXT,
                UNIQUE(category, key, model)
            )
        """)

        self._conn.execute("""
            INSERT INTO memories
            SELECT id, content, memory_type, category, key, tags, confidence,
                   source, created_at, updated_at, accessed_at, access_count,
                   COALESCE(model, 'legacy'), context
            FROM memories_old
        """)

        self._conn.execute("DROP TABLE memories_old")

        # 4. Rebuild FTS triggers
        self._conn.execute("""
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, category, key, tags)
                VALUES (new.rowid, new.content, new.category, new.key, new.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
                INSERT INTO memories_fts(rowid, content, category, key, tags)
                VALUES (new.rowid, new.content, new.category, new.key, new.tags);
            END
        """)

        # 5. Rebuild FTS content to match new rowids
        self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

        # 6. Rebuild indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(category, key)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_model ON memories(model)")

        # 7. Rebuild embeddings table to fix FK reference after table rebuild
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_new (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        existing_embeds = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'embeddings'"
        ).fetchone()
        if existing_embeds:
            self._conn.execute("INSERT OR IGNORE INTO embeddings_new SELECT * FROM embeddings")
            self._conn.execute("DROP TABLE embeddings")
        self._conn.execute("ALTER TABLE embeddings_new RENAME TO embeddings")

        # 8. Record migration
        self._conn.execute(
            "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            (2, _now(), "Per-model memory partitioning: UNIQUE(category,key,model), model NOT NULL"),
        )
        self._conn.commit()

        # Re-enable FK checks
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info("Schema migration v2 complete")

    def _migrate_v3(self):
        """v3: Challenge/debate system — challenges + challenge_arguments tables."""
        logger.info("Running schema migration v3: challenge/debate system")

        # Tables are created by SCHEMA if they don't exist, but we still need
        # to record the migration version for idempotency.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            )
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            (3, _now(), "Challenge/debate system: challenges + challenge_arguments tables"),
        )
        self._conn.commit()
        logger.info("Schema migration v3 complete")

    def _migrate_v4(self):
        """v4: Add host column to memories for per-system provenance."""
        logger.info("Running schema migration v4: host column")

        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "host" not in existing:
            self._conn.execute("ALTER TABLE memories ADD COLUMN host TEXT")

        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_host ON memories(host)")

        self._conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            (4, _now(), "Add host column to memories for per-system provenance"),
        )
        self._conn.commit()
        logger.info("Schema migration v4 complete")

    def _migrate_v5(self):
        """v5: Add 'diary' memory type, migrate existing diary entries."""
        logger.info("Running schema migration v5: diary memory type")

        self._conn.execute("PRAGMA foreign_keys=OFF")

        # 1. Rebuild memories table with updated CHECK constraint
        self._conn.execute("DROP TRIGGER IF EXISTS memories_ai")
        self._conn.execute("DROP TRIGGER IF EXISTS memories_ad")
        self._conn.execute("DROP TRIGGER IF EXISTS memories_au")

        self._conn.execute("ALTER TABLE memories RENAME TO memories_old")

        self._conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('fact', 'episode', 'preference', 'diary')),
                category TEXT NOT NULL DEFAULT 'general',
                key TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT 'legacy',
                context TEXT,
                host TEXT,
                UNIQUE(category, key, model)
            )
        """)

        self._conn.execute("""
            INSERT INTO memories
            SELECT id, content, memory_type, category, key, tags, confidence,
                   source, created_at, updated_at, accessed_at, access_count,
                   model, context, host
            FROM memories_old
        """)

        self._conn.execute("DROP TABLE memories_old")

        # 2. Convert existing diary entries: episodes in category 'claude' with 'diary' tag
        self._conn.execute("""
            UPDATE memories SET memory_type = 'diary'
            WHERE memory_type = 'episode'
              AND category = 'claude'
              AND tags LIKE '%diary%'
        """)

        # 3. Rebuild FTS triggers
        self._conn.execute("""
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, category, key, tags)
                VALUES (new.rowid, new.content, new.category, new.key, new.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, category, key, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.key, old.tags);
                INSERT INTO memories_fts(rowid, content, category, key, tags)
                VALUES (new.rowid, new.content, new.category, new.key, new.tags);
            END
        """)

        # 4. Rebuild FTS content
        self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

        # 5. Rebuild indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(category, key)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_model ON memories(model)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_host ON memories(host)")

        # 6. Rebuild embeddings FK reference
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_new (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        existing_embeds = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'embeddings'"
        ).fetchone()
        if existing_embeds:
            self._conn.execute("INSERT OR IGNORE INTO embeddings_new SELECT * FROM embeddings")
            self._conn.execute("DROP TABLE embeddings")
        self._conn.execute("ALTER TABLE embeddings_new RENAME TO embeddings")

        # 7. Record migration
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            (5, _now(), "Add diary memory type, migrate existing diary entries"),
        )
        self._conn.commit()

        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info("Schema migration v5 complete")

    def close(self):
        if self._embed_client:
            self._embed_client.close()
        self._conn.close()

    # ── Embedding helpers ─────────────────────────────────────────

    def _embed(self, text: str) -> list[float] | None:
        """Generate embedding via Ollama. Returns None on failure."""
        try:
            import httpx
        except ImportError:
            return None
        try:
            if self._embed_client is None:
                self._embed_client = httpx.Client(timeout=30.0)
            resp = self._embed_client.post(
                self._ollama_url,
                json={"model": EMBED_MODEL, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"embeddings": [[...]]} for single input
            embeddings = data.get("embeddings") or data.get("embedding")
            if embeddings:
                vec = embeddings[0] if isinstance(embeddings[0], list) else embeddings
                if len(vec) == EMBED_DIM:
                    return vec
            return None
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return None

    def _serialize_vec(self, vec: list[float]) -> bytes:
        """Serialize float list to little-endian f32 bytes for sqlite-vec."""
        return struct.pack(f"<{len(vec)}f", *vec)

    def _store_embedding(self, memory_id: str, vec: list[float]):
        """Store embedding in both the embeddings table and vec_memories."""
        now = _now()
        blob = self._serialize_vec(vec)

        # Upsert into embeddings table (metadata + raw blob)
        self._conn.execute(
            """INSERT INTO embeddings (memory_id, embedding, model, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(memory_id) DO UPDATE SET
               embedding = excluded.embedding, model = excluded.model, created_at = excluded.created_at""",
            (memory_id, blob, EMBED_MODEL, now),
        )

        # Upsert into vec_memories (KNN search table)
        if self._vec_available:
            # vec0 doesn't support ON CONFLICT — delete then insert
            self._conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
            self._conn.execute(
                "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)",
                (memory_id, blob),
            )

    def _embed_and_store(self, memory_id: str, content: str):
        """Generate and store embedding for a memory. No-op on failure."""
        vec = self._embed(content)
        if vec:
            self._store_embedding(memory_id, vec)
            self._conn.commit()

    def _delete_embedding(self, memory_id: str):
        """Remove embedding for a deleted memory."""
        self._conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
        if self._vec_available:
            self._conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))

    def _find_similar_same_model(
        self, vec: list[float], model: str, threshold: float = DUPLICATE_DISTANCE_THRESHOLD
    ) -> tuple["Memory", float] | None:
        """Find the most similar memory from the same model family.

        Returns (existing_memory, distance) if a near-duplicate exists, else None.
        Only searches within the caller's model family — cross-model similarities are ignored.
        """
        if not self._vec_available:
            return None
        blob = self._serialize_vec(vec)
        caller_fam = model_family(model)
        if not caller_fam:
            return None
        try:
            rows = self._conn.execute(
                "SELECT memory_id, distance FROM vec_memories "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, 20),
            ).fetchall()
        except Exception as e:
            logger.debug(f"Duplicate check vec search failed: {e}")
            return None

        for row in rows:
            if row["distance"] >= threshold:
                break  # sorted by distance ascending, no more matches possible
            mem_row = self._conn.execute(
                "SELECT model FROM memories WHERE id = ?", (row["memory_id"],)
            ).fetchone()
            if mem_row and model_family(mem_row["model"]) == caller_fam:
                return (self._get_memory(row["memory_id"]), row["distance"])
        return None

    @staticmethod
    def _activation(access_count: int, days_since_last_access: float, decay: float = ACTR_DECAY) -> float:
        """ACT-R base-level activation: B_i ≈ ln(n+1) - d·ln(t+1).

        Higher values = more active/remembered. Negative values = effectively forgotten.
        - access_count: total number of times the memory has been retrieved
        - days_since_last_access: days since last access (clamped to >= 0.001)
        - decay: power-law decay rate (0.5 = standard ACT-R)
        """
        t = max(days_since_last_access, 0.001)
        return math.log(access_count + 1) - decay * math.log(t + 1)

    @staticmethod
    def _rrf_fuse(
        fts_ids: list[str],
        vec_ids: list[str],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of two ranked lists. Returns (id, score) sorted desc."""
        scores: dict[str, float] = {}
        for rank, mid in enumerate(fts_ids):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
        for rank, mid in enumerate(vec_ids):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _vec_search(
        self, query: str, limit: int,
        before: str | None = None, after: str | None = None,
    ) -> list[tuple[str, float]]:
        """KNN vector search. Returns (memory_id, distance) pairs ranked by similarity."""
        if not self._vec_available:
            return []
        vec = self._embed(query)
        if vec is None:
            return []
        blob = self._serialize_vec(vec)
        try:
            # Overfetch if temporal filtering will remove some results
            fetch_limit = limit * 3 if (before or after) else limit
            rows = self._conn.execute(
                "SELECT memory_id, distance FROM vec_memories WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (blob, fetch_limit),
            ).fetchall()
            results = [(r["memory_id"], r["distance"]) for r in rows]
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
            return []

        # Post-filter by created_at if temporal bounds provided
        if results and (before or after):
            ids = [mid for mid, _ in results]
            placeholders = ",".join("?" * len(ids))
            filter_sql = f"SELECT id FROM memories WHERE id IN ({placeholders})"
            filter_params: list = list(ids)
            if after:
                filter_sql += " AND created_at >= ?"
                filter_params.append(after)
            if before:
                filter_sql += " AND created_at <= ?"
                filter_params.append(before)
            valid = {r["id"] for r in self._conn.execute(filter_sql, filter_params).fetchall()}
            results = [(mid, dist) for mid, dist in results if mid in valid]

        return results[:limit]

    # ── Remember ────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        category: str = "general",
        key: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        model: str = "legacy",
        context: str | None = None,
        host: str | None = None,
    ) -> Memory:
        """Store a new memory. If a memory with the same category+key+model exists, update it.

        Raises DuplicateMemoryError if a semantically similar memory from the same model
        family already exists (facts and preferences only, not episodes).
        """
        now = _now()
        tags = tags or []
        mid = _short_id(content)

        if key:
            # Upsert: update existing memory with same category+key+model
            existing = self._conn.execute(
                "SELECT * FROM memories WHERE category = ? AND key = ? AND model = ?",
                (category, key, model),
            ).fetchone()
            if existing:
                # Record history before overwriting
                self._record_history(
                    memory_id=existing["id"],
                    action="updated",
                    model=model,
                    old_content=existing["content"],
                    new_content=content,
                    old_confidence=existing["confidence"],
                    new_confidence=confidence,
                    context=context,
                )
                self._conn.execute(
                    """UPDATE memories SET content = ?, tags = ?, confidence = ?,
                       source = ?, updated_at = ?, accessed_at = ?,
                       model = ?, context = ?, host = ?
                       WHERE id = ?""",
                    (content, json.dumps(tags), confidence, source, now, now,
                     model, context, host, existing["id"]),
                )
                self._conn.commit()
                self._embed_and_store(existing["id"], content)
                return self._get_memory(existing["id"])

        # Embed content early for duplicate detection
        vec = self._embed(content)

        # Duplicate detection: check for same-model semantic near-duplicates.
        # Only for facts and preferences — episodes are unique events by definition.
        # Skip for short content (<50 chars) — embeddings aren't discriminative enough.
        if vec and self._vec_available and memory_type not in ("episode", "diary") and len(content) >= 50:
            similar = self._find_similar_same_model(vec, model)
            if similar:
                raise DuplicateMemoryError(similar[0], similar[1])

        try:
            self._conn.execute(
                """INSERT INTO memories (id, content, memory_type, category, key, tags,
                   confidence, source, created_at, updated_at, accessed_at, access_count,
                   model, context, host)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (mid, content, memory_type, category, key, json.dumps(tags),
                 confidence, source, now, now, now, model, context, host),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # ID collision (extremely unlikely) — retry with different salt
            mid = _short_id(content, salt="retry")
            self._conn.execute(
                """INSERT INTO memories (id, content, memory_type, category, key, tags,
                   confidence, source, created_at, updated_at, accessed_at, access_count,
                   model, context, host)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (mid, content, memory_type, category, key, json.dumps(tags),
                 confidence, source, now, now, now, model, context, host),
            )
            self._conn.commit()

        # Record creation in history
        self._record_history(
            memory_id=mid,
            action="created",
            model=model,
            old_content=None,
            new_content=content,
            old_confidence=None,
            new_confidence=confidence,
            context=context,
        )

        # Store embedding (reuse the one we already computed)
        if vec:
            self._store_embedding(mid, vec)
            self._conn.commit()
        else:
            self._embed_and_store(mid, content)
        return self._get_memory(mid)

    # ── Recall ──────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        category: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.0,
        caller_model: str | None = None,
        scope: str = "all",
        host: str | None = None,
        caller_host: str | None = None,
        before: str | None = None,
        after: str | None = None,
        min_similarity: float | None = None,
    ) -> list[Memory] | RecallResult:
        """Hybrid search: FTS5 keyword + vector KNN, fused with RRF.

        scope: "all" (default) returns all models' memories; "own" filters to caller's model family only.
        host: if provided, filter to memories with exactly this host (or NULL host — host-agnostic facts always included).
        caller_host: if provided, boost memories matching this host and penalize mismatched hosts.
        before: ISO 8601 date — only return memories created before this date.
        after: ISO 8601 date — only return memories created after this date.
        min_similarity: L2 distance threshold for abstention. If no memory is closer than
            this threshold, returns empty results instead of irrelevant matches.
        """
        overfetch = limit * 3
        caller_fam = model_family(caller_model) if caller_model else None
        scope_own = scope == "own" and caller_fam is not None

        # ── Path A: FTS5 keyword search ──
        fts_query = self._build_fts_query(query)
        sql = """
            SELECT m.*, fts.rank AS fts_rank
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
        """
        params: list = [fts_query]

        if scope_own:
            sql += " AND m.model LIKE ?"
            params.append(f"{caller_fam}%")
        if category:
            sql += " AND m.category = ?"
            params.append(category)
        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type)
        if min_confidence > 0:
            sql += " AND m.confidence >= ?"
            params.append(min_confidence)
        if host is not None:
            sql += " AND (m.host = ? OR m.host IS NULL)"
            params.append(host)
        if after:
            sql += " AND m.created_at >= ?"
            params.append(after)
        if before:
            sql += " AND m.created_at <= ?"
            params.append(before)

        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(overfetch)

        fts_rows = self._conn.execute(sql, params).fetchall()
        fts_ids = [r["id"] for r in fts_rows]

        # ── Path B: Vector KNN search ──
        vec_results = self._vec_search(query, overfetch, before=before, after=after)
        vec_ids = [vid for vid, _ in vec_results]
        vec_distances = {vid: dist for vid, dist in vec_results}

        # Abstention check: if best vector match is too distant, nothing relevant exists
        if min_similarity is not None and vec_distances:
            best_distance = min(vec_distances.values())
            if best_distance > min_similarity:
                if caller_model is not None:
                    return RecallResult(own=[], others=[], unknown=[])
                return []

        # Post-filter vec results for scope="own" (vec0 doesn't support WHERE)
        if scope_own and vec_ids:
            owned = set()
            placeholders = ",".join("?" for _ in vec_ids)
            rows = self._conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) AND model LIKE ?",
                vec_ids + [f"{caller_fam}%"],
            ).fetchall()
            owned = {r["id"] for r in rows}
            vec_ids = [mid for mid in vec_ids if mid in owned]

        # ── Fusion ──
        if vec_ids:
            # RRF merge of both ranked lists
            fused = self._rrf_fuse(fts_ids, vec_ids)
            candidate_ids = [mid for mid, _ in fused]
        else:
            # Fallback: FTS-only (Ollama down or no embeddings)
            candidate_ids = fts_ids

        if not candidate_ids:
            return []

        # Fetch full Memory objects for candidates
        placeholders = ",".join("?" for _ in candidate_ids)
        filter_clauses = ""
        filter_params: list = list(candidate_ids)
        if category:
            filter_clauses += " AND category = ?"
            filter_params.append(category)
        if memory_type:
            filter_clauses += " AND memory_type = ?"
            filter_params.append(memory_type)
        if min_confidence > 0:
            filter_clauses += " AND confidence >= ?"
            filter_params.append(min_confidence)
        if host is not None:
            filter_clauses += " AND (host = ? OR host IS NULL)"
            filter_params.append(host)

        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}){filter_clauses}",
            filter_params,
        ).fetchall()
        mem_by_id = {r["id"]: r for r in rows}

        # Build ordered list preserving RRF/FTS rank
        memories = []
        for mid in candidate_ids:
            if mid in mem_by_id:
                memories.append(self._row_to_memory(mem_by_id[mid]))

        # Tag filter (post-query since tags are JSON)
        if tags:
            tag_set = set(tags)
            memories = [m for m in memories if tag_set & set(m.tags)]

        # Compute final scores: RRF rank is primary, composite is tiebreaker
        rrf_scores = {mid: score for mid, score in fused} if vec_ids else {}
        for m in memories:
            composite = self._compute_score(m, fts_rows)
            rrf = rrf_scores.get(m.id, 0.0)
            # RRF dominates (scaled up); composite is tiebreaker
            m.score = rrf * 100.0 + composite
            # Host-based provenance boost: match = +2.0, mismatch = -5.0, NULL = 0
            if caller_host:
                if m.host == caller_host:
                    m.score += 2.0
                elif m.host is not None:
                    m.score -= 5.0

        memories.sort(key=lambda m: m.score, reverse=True)
        memories = memories[:limit]

        # Update access timestamps
        now = _now()
        for m in memories:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (now, m.id),
            )
        self._conn.commit()

        if caller_model is None:
            return memories  # backward compat: flat list

        own, others, unknown = [], [], []
        for m in memories:
            mem_fam = model_family(m.model)
            if mem_fam is None:
                unknown.append(m)
            elif caller_fam and mem_fam == caller_fam:
                own.append(m)
            else:
                others.append(m)
        return RecallResult(own=own, others=others, unknown=unknown)

    def recall_by_id(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        now = _now()
        self._conn.execute(
            "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()
        return self._row_to_memory(row)

    # ── Forget ──────────────────────────────────────────────────────

    def forget(self, memory_id: str, model: str = "legacy", context: str | None = None) -> bool:
        """Delete a memory by ID. Model must own the memory."""
        existing = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not existing:
            return False
        existing_fam = model_family(existing["model"])
        caller_fam = model_family(model)
        if existing_fam and caller_fam and existing_fam != caller_fam:
            raise PermissionError(
                f"Cannot forget memory owned by {existing['model']} (family '{existing_fam}'). "
                f"You are {model} (family '{caller_fam}'). Only the owning model family can forget."
            )
        self._record_history(
            memory_id=memory_id,
            action="forgotten",
            model=model,
            old_content=existing["content"],
            new_content=None,
            old_confidence=existing["confidence"],
            new_confidence=None,
            context=context,
        )
        self._delete_embedding(memory_id)
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return True

    def forget_by_key(self, category: str, key: str, model: str = "legacy", context: str | None = None) -> bool:
        """Delete a fact by category + key + model. Model-scoped."""
        existing = self._conn.execute(
            "SELECT * FROM memories WHERE category = ? AND key = ? AND model = ?", (category, key, model)
        ).fetchone()
        if not existing:
            return False
        self._record_history(
            memory_id=existing["id"],
            action="forgotten",
            model=model,
            old_content=existing["content"],
            new_content=None,
            old_confidence=existing["confidence"],
            new_confidence=None,
            context=context,
        )
        self._delete_embedding(existing["id"])
        self._conn.execute(
            "DELETE FROM memories WHERE category = ? AND key = ? AND model = ?",
            (category, key, model),
        )
        self._conn.commit()
        return True

    # ── Update ──────────────────────────────────────────────────────

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        key: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        model: str = "legacy",
        context: str | None = None,
        host: str | None = None,
    ) -> Memory | None:
        """Update fields of an existing memory. Model must own the memory."""
        existing = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not existing:
            return None
        existing_fam = model_family(existing["model"])
        caller_fam = model_family(model)
        if existing_fam and caller_fam and existing_fam != caller_fam:
            raise PermissionError(
                f"Cannot update memory owned by {existing['model']} (family '{existing_fam}'). "
                f"You are {model} (family '{caller_fam}'). Only the owning model family can update."
            )

        # Record history
        self._record_history(
            memory_id=memory_id,
            action="updated",
            model=model,
            old_content=existing["content"],
            new_content=content or existing["content"],
            old_confidence=existing["confidence"],
            new_confidence=confidence if confidence is not None else existing["confidence"],
            context=context,
        )

        now = _now()
        updates = {"updated_at": now}
        if content is not None:
            updates["content"] = content
        if category is not None:
            updates["category"] = category
        if key is not None:
            updates["key"] = key
        if tags is not None:
            updates["tags"] = json.dumps(tags)
        if confidence is not None:
            updates["confidence"] = confidence
        if model is not None:
            updates["model"] = model
        if context is not None:
            updates["context"] = context
        if host is not None:
            updates["host"] = host

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id]
        self._conn.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

        # Re-embed if content changed
        if content is not None:
            self._embed_and_store(memory_id, content)

        return self._get_memory(memory_id)

    # ── Relationships ───────────────────────────────────────────────

    def relate(
        self,
        entity_from: str,
        entity_to: str,
        relation_type: str,
        metadata: dict | None = None,
        confidence: float = 1.0,
    ) -> Relationship:
        """Create or update a relationship between two entities."""
        now = _now()
        rid = _short_id(f"{entity_from}:{entity_to}:{relation_type}")

        existing = self._conn.execute(
            "SELECT id FROM relationships WHERE entity_from = ? AND entity_to = ? AND relation_type = ?",
            (entity_from, entity_to, relation_type),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE relationships SET metadata = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata) if metadata else None, confidence, now, existing["id"]),
            )
            self._conn.commit()
            rid = existing["id"]
        else:
            self._conn.execute(
                """INSERT INTO relationships (id, entity_from, entity_to, relation_type,
                   metadata, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (rid, entity_from, entity_to, relation_type,
                 json.dumps(metadata) if metadata else None, confidence, now, now),
            )
            self._conn.commit()

        return self._get_relationship(rid)

    def get_entity(self, entity: str) -> dict:
        """Get all memories and relationships for an entity."""
        # Search memories mentioning this entity
        memories = self.recall(entity, limit=20)

        # Get relationships
        rows = self._conn.execute(
            """SELECT * FROM relationships
               WHERE entity_from = ? OR entity_to = ?
               ORDER BY confidence DESC""",
            (entity, entity),
        ).fetchall()
        relationships = [self._row_to_relationship(r) for r in rows]

        return {
            "entity": entity,
            "memories": [m.to_dict() for m in memories],
            "relationships": [r.to_dict() for r in relationships],
        }

    def unrelate(self, entity_from: str, entity_to: str, relation_type: str) -> bool:
        """Remove a relationship."""
        cur = self._conn.execute(
            "DELETE FROM relationships WHERE entity_from = ? AND entity_to = ? AND relation_type = ?",
            (entity_from, entity_to, relation_type),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Introspection ───────────────────────────────────────────────

    def list_categories(self) -> list[dict]:
        """List all categories with memory counts."""
        rows = self._conn.execute(
            "SELECT category, memory_type, COUNT(*) as count FROM memories GROUP BY category, memory_type ORDER BY count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Get overall memory statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = self._conn.execute(
            "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
        ).fetchall()
        relationships = self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        oldest = self._conn.execute("SELECT MIN(created_at) FROM memories").fetchone()[0]
        newest = self._conn.execute("SELECT MAX(updated_at) FROM memories").fetchone()[0]
        stale = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE accessed_at < datetime('now', '-30 days')"
        ).fetchone()[0]
        embedded = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

        by_model = self._conn.execute(
            "SELECT model, COUNT(*) as count FROM memories GROUP BY model ORDER BY count DESC"
        ).fetchall()

        return {
            "total_memories": total,
            "by_type": {r["memory_type"]: r["count"] for r in by_type},
            "by_model": {r["model"]: r["count"] for r in by_model},
            "total_relationships": relationships,
            "oldest_memory": oldest,
            "newest_memory": newest,
            "stale_memories_30d": stale,
            "embedded_memories": embedded,
            "embed_model": EMBED_MODEL,
            "vec_search_available": self._vec_available,
            "db_path": str(self.db_path),
        }

    def list_models(self) -> list[dict]:
        """List all models with memory counts and latest activity."""
        rows = self._conn.execute("""
            SELECT model, COUNT(*) as count,
                   MAX(updated_at) as latest_update,
                   MIN(created_at) as first_memory
            FROM memories
            GROUP BY model
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def find_disagreements(self) -> list[dict]:
        """Find category+key pairs where multiple models have different memories.

        Returns rows with category, key, model_count (number of distinct models).
        """
        rows = self._conn.execute("""
            SELECT category, key, COUNT(DISTINCT model) as model_count,
                   GROUP_CONCAT(model, ', ') as models
            FROM memories
            WHERE key IS NOT NULL
            GROUP BY category, key
            HAVING model_count > 1
            ORDER BY model_count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_stale(self, days: int = 30, limit: int = 20) -> list[Memory]:
        """Get least-active memories using ACT-R power-law decay.

        Pre-filters to memories not accessed in `days` days, then ranks by
        activation score (lowest first = most forgotten). Activation uses:
            B_i = ln(access_count + 1) - 0.5 * ln(days_since_last_access + 1)
        """
        rows = self._conn.execute(
            """SELECT * FROM memories
               WHERE accessed_at < datetime('now', ? || ' days')""",
            (f"-{days}",),
        ).fetchall()
        now = datetime.now(timezone.utc)
        scored = []
        for row in rows:
            mem = self._row_to_memory(row)
            try:
                last_access = datetime.fromisoformat(mem.accessed_at)
                days_since = max((now - last_access).total_seconds() / 86400, 0.001)
            except (ValueError, TypeError):
                days_since = 999.0
            mem.score = self._activation(mem.access_count, days_since)
            scored.append(mem)
        scored.sort(key=lambda m: m.score)  # lowest activation = most stale
        return scored[:limit]

    # ── Challenges / Debate ────────────────────────────────────────

    def challenge(
        self,
        target_memory_id: str,
        model: str,
        argument: str,
        evidence: list[str] | None = None,
    ) -> Challenge:
        """Start a new challenge against an existing memory.

        The challenger argues the target memory is wrong/incomplete and provides
        their reasoning. Other models can then respond via debate().
        """
        # Verify target memory exists
        target = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (target_memory_id,)
        ).fetchone()
        if not target:
            raise ValueError(f"Target memory {target_memory_id} not found")

        # Check for existing open challenge on this memory by this model family
        caller_fam = model_family(model)
        existing = self._conn.execute(
            "SELECT c.* FROM challenges c WHERE c.target_memory_id = ? AND c.status = 'open'",
            (target_memory_id,),
        ).fetchall()
        for row in existing:
            if model_family(row["challenger_model"]) == caller_fam:
                raise ValueError(
                    f"Open challenge already exists from {row['challenger_model']} "
                    f"on this memory [{row['id']}]. Use 'debate' to add arguments."
                )

        now = _now()
        cid = _short_id(f"challenge:{target_memory_id}:{model}")
        aid = _short_id(f"arg:{cid}:{model}")

        self._conn.execute(
            """INSERT INTO challenges (id, target_memory_id, challenger_model, status, created_at)
               VALUES (?, ?, ?, 'open', ?)""",
            (cid, target_memory_id, model, now),
        )
        self._conn.execute(
            """INSERT INTO challenge_arguments (id, challenge_id, model, position, argument, evidence, created_at)
               VALUES (?, ?, ?, 'challenge', ?, ?, ?)""",
            (aid, cid, model, argument, json.dumps(evidence or []), now),
        )
        self._conn.commit()
        return self._get_challenge(cid)

    def debate(
        self,
        challenge_id: str,
        model: str,
        argument: str,
        position: str = "rebuttal",
        evidence: list[str] | None = None,
    ) -> Challenge:
        """Add an argument to an existing challenge. Any model can participate.

        position: 'defense' (supports target memory), 'rebuttal' (supports challenger),
                  'synthesis' (proposes a merged view).
        """
        challenge = self._conn.execute(
            "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
        ).fetchone()
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")
        if challenge["status"] != "open":
            raise ValueError(f"Challenge {challenge_id} is already {challenge['status']}")
        if position not in ("defense", "rebuttal", "synthesis"):
            raise ValueError(f"Position must be 'defense', 'rebuttal', or 'synthesis'")

        now = _now()
        aid = _short_id(f"arg:{challenge_id}:{model}")

        self._conn.execute(
            """INSERT INTO challenge_arguments (id, challenge_id, model, position, argument, evidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (aid, challenge_id, model, position, argument, json.dumps(evidence or []), now),
        )
        self._conn.commit()
        return self._get_challenge(challenge_id)

    def get_challenges(
        self,
        status: str | None = "open",
        target_memory_id: str | None = None,
        limit: int = 20,
    ) -> list[Challenge]:
        """List challenges, optionally filtered by status and/or target memory."""
        sql = "SELECT * FROM challenges WHERE 1=1"
        params: list = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if target_memory_id:
            sql += " AND target_memory_id = ?"
            params.append(target_memory_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._get_challenge(r["id"]) for r in rows]

    def resolve_challenge(
        self,
        challenge_id: str,
        status: str,
        resolution: str,
        resolved_by: str,
    ) -> Challenge:
        """Resolve a challenge.

        status: 'accepted' (target was wrong), 'rejected' (challenge was wrong),
                'synthesized' (both had partial truth — create a new memory with the merged view).
        """
        challenge = self._conn.execute(
            "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
        ).fetchone()
        if not challenge:
            raise ValueError(f"Challenge {challenge_id} not found")
        if challenge["status"] != "open":
            raise ValueError(f"Challenge {challenge_id} is already {challenge['status']}")
        if status not in ("accepted", "rejected", "synthesized"):
            raise ValueError(f"Resolution status must be 'accepted', 'rejected', or 'synthesized'")

        now = _now()
        self._conn.execute(
            "UPDATE challenges SET status = ?, resolution = ?, resolved_by = ?, resolved_at = ? WHERE id = ?",
            (status, resolution, resolved_by, now, challenge_id),
        )
        self._conn.commit()
        return self._get_challenge(challenge_id)

    def _get_challenge(self, challenge_id: str) -> Challenge:
        """Load a challenge with all its arguments."""
        row = self._conn.execute(
            "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Challenge {challenge_id} not found")

        arg_rows = self._conn.execute(
            "SELECT * FROM challenge_arguments WHERE challenge_id = ? ORDER BY created_at ASC",
            (challenge_id,),
        ).fetchall()
        arguments = [
            ChallengeArgument(
                id=a["id"],
                challenge_id=a["challenge_id"],
                model=a["model"],
                position=a["position"],
                argument=a["argument"],
                evidence=json.loads(a["evidence"]),
                created_at=a["created_at"],
            )
            for a in arg_rows
        ]
        return Challenge(
            id=row["id"],
            target_memory_id=row["target_memory_id"],
            challenger_model=row["challenger_model"],
            status=row["status"],
            resolution=row["resolution"],
            resolved_by=row["resolved_by"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
            arguments=arguments,
        )

    # ── Backfill ─────────────────────────────────────────────────────

    def backfill_embeddings(self, batch_size: int = 50) -> dict:
        """Embed all memories that don't yet have vectors. Returns progress summary."""
        # Find memories missing from vec_memories (or embeddings table if vec not available)
        if self._vec_available:
            rows = self._conn.execute(
                """SELECT m.id, m.content FROM memories m
                   LEFT JOIN vec_memories v ON m.id = v.memory_id
                   WHERE v.memory_id IS NULL"""
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT m.id, m.content FROM memories m
                   LEFT JOIN embeddings e ON m.id = e.memory_id
                   WHERE e.memory_id IS NULL"""
            ).fetchall()

        total = len(rows)
        embedded = 0
        failed = 0
        stale_model = 0

        # Also find embeddings with a different model (stale)
        stale_rows = self._conn.execute(
            "SELECT memory_id FROM embeddings WHERE model != ?", (EMBED_MODEL,)
        ).fetchall()
        stale_ids = {r["memory_id"] for r in stale_rows}

        # Re-embed stale ones too
        if stale_ids:
            stale_mem_rows = self._conn.execute(
                f"SELECT id, content FROM memories WHERE id IN ({','.join('?' for _ in stale_ids)})",
                list(stale_ids),
            ).fetchall()
            rows = list(rows) + list(stale_mem_rows)
            stale_model = len(stale_ids)

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            for row in batch:
                vec = self._embed(row["content"])
                if vec:
                    self._store_embedding(row["id"], vec)
                    embedded += 1
                else:
                    failed += 1
            self._conn.commit()

        return {
            "total": total + stale_model,
            "embedded": embedded,
            "failed": failed,
            "stale_model_reembedded": min(stale_model, embedded),
        }

    # ── Context (priming) ───────────────────────────────────────────

    def get_context(self, topic: str | None = None, limit: int = 20, caller_model: str | None = None, caller_host: str | None = None) -> dict:
        """Get a priming context for conversation start."""
        result = {"preferences": [], "recent": [], "topic_memories": []}
        caller_fam = model_family(caller_model) if caller_model else None

        def _annotate(mem_dict: dict) -> dict:
            """Add provenance field when caller_model is provided."""
            if caller_fam is None:
                return mem_dict
            mem_fam = model_family(mem_dict.get("model"))
            if mem_fam is None:
                mem_dict["provenance"] = "unknown"
            elif mem_fam == caller_fam:
                mem_dict["provenance"] = "own"
            else:
                mem_dict["provenance"] = "other"
            return mem_dict

        # Always include preferences
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE memory_type = 'preference' ORDER BY confidence DESC LIMIT 10"
        ).fetchall()
        result["preferences"] = [_annotate(self._row_to_memory(r).to_dict()) for r in rows]

        # Recently updated facts
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE memory_type = 'fact' ORDER BY updated_at DESC LIMIT 10"
        ).fetchall()
        result["recent"] = [_annotate(self._row_to_memory(r).to_dict()) for r in rows]

        # Topic-specific if provided
        if topic:
            memories = self.recall(topic, limit=limit, caller_model=caller_model, caller_host=caller_host)
            if isinstance(memories, RecallResult):
                result["topic_memories"] = [_annotate(m.to_dict()) for m in memories.all]
            else:
                result["topic_memories"] = [_annotate(m.to_dict()) for m in memories]

        return result

    # ── History / Provenance ────────────────────────────────────────

    def get_history(self, memory_id: str) -> list[HistoryEntry]:
        """Get the full audit trail for a memory."""
        rows = self._conn.execute(
            "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY created_at ASC",
            (memory_id,),
        ).fetchall()
        return [self._row_to_history(r) for r in rows]

    def get_history_by_model(self, model: str, limit: int = 20) -> list[HistoryEntry]:
        """Get all history entries from a specific model."""
        rows = self._conn.execute(
            "SELECT * FROM memory_history WHERE model = ? ORDER BY created_at DESC LIMIT ?",
            (model, limit),
        ).fetchall()
        return [self._row_to_history(r) for r in rows]

    def _record_history(
        self,
        memory_id: str,
        action: str,
        model: str | None,
        old_content: str | None,
        new_content: str | None,
        old_confidence: float | None,
        new_confidence: float | None,
        context: str | None,
    ):
        """Record a history entry for a memory mutation."""
        hid = _short_id(f"{memory_id}:{action}")
        now = _now()
        self._conn.execute(
            """INSERT INTO memory_history (id, memory_id, action, model,
               old_content, new_content, old_confidence, new_confidence,
               context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (hid, memory_id, action, model, old_content, new_content,
             old_confidence, new_confidence, context, now),
        )
        # Don't commit here — caller manages the transaction

    def _row_to_history(self, row: sqlite3.Row) -> HistoryEntry:
        return HistoryEntry(
            id=row["id"],
            memory_id=row["memory_id"],
            action=row["action"],
            model=row["model"],
            old_content=row["old_content"],
            new_content=row["new_content"],
            old_confidence=row["old_confidence"],
            new_confidence=row["new_confidence"],
            context=row["context"],
            created_at=row["created_at"],
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _get_memory(self, mid: str) -> Memory:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        return self._row_to_memory(row)

    def _get_relationship(self, rid: str) -> Relationship:
        row = self._conn.execute("SELECT * FROM relationships WHERE id = ?", (rid,)).fetchone()
        return self._row_to_relationship(row)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            category=row["category"],
            key=row["key"],
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
            model=row["model"],
            context=row["context"],
            host=row["host"] if "host" in row.keys() else None,
        )

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            entity_from=row["entity_from"],
            entity_to=row["entity_to"],
            relation_type=row["relation_type"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _build_fts_query(self, query: str) -> str:
        """Build an FTS5 query from natural language input.

        Tokenizes, strips FTS operators, and joins with OR for broad matching.
        """
        # Split into words, strip punctuation, remove FTS5 operators
        fts_operators = {"AND", "OR", "NOT", "NEAR"}
        words = []
        for word in query.split():
            cleaned = word.strip('".,:;!?()[]{}')
            if cleaned and cleaned.upper() not in fts_operators and len(cleaned) > 1:
                words.append(f'"{cleaned}"')
        if not words:
            return f'"{query}"'
        return " OR ".join(words)

    def _compute_score(self, memory: Memory, raw_rows: list[sqlite3.Row]) -> float:
        """Composite relevance score: FTS rank + recency + frequency + confidence."""
        # Find the raw FTS rank for this memory
        fts_rank = 0.0
        for row in raw_rows:
            if row["id"] == memory.id:
                fts_rank = abs(row["fts_rank"])  # FTS5 rank is negative
                break

        # Normalize FTS rank (higher = better)
        fts_score = fts_rank

        # Recency: exponential decay from last update
        try:
            updated = datetime.fromisoformat(memory.updated_at)
            age_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400
            recency_score = math.exp(-0.02 * age_days)  # half-life ~35 days
        except (ValueError, TypeError):
            recency_score = 0.5

        # Frequency: log scale of access count
        freq_score = math.log1p(memory.access_count) / 10.0

        # Composite: weighted sum
        return (fts_score * 0.5) + (recency_score * 0.3) + (freq_score * 0.1) + (memory.confidence * 0.1)
