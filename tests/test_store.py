"""Tests for the engram memory store."""

import os
import tempfile
import pytest
from engram.store import MemoryStore, RecallResult, DuplicateMemoryError, Challenge, model_family


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = MemoryStore(path)
    yield s
    s.close()
    os.unlink(path)


# ── Basic CRUD ──────────────────────────────────────────────────


def test_remember_and_recall(store):
    m = store.remember("TrueNAS is at 192.168.0.192", category="network", key="truenas_ip", model="claude-opus-4-6")
    assert m.id
    assert m.content == "TrueNAS is at 192.168.0.192"
    assert m.category == "network"

    results = store.recall("TrueNAS IP address")
    assert len(results) >= 1
    assert any("192.168.0.192" in r.content for r in results)


def test_fact_deduplication_same_model(store):
    """Same model writing same category+key = upsert."""
    store.remember("NAS is at .192", category="network", key="nas_ip", model="claude-opus-4-6")
    store.remember("NAS is at .193", category="network", key="nas_ip", model="claude-opus-4-6")

    results = store.recall("NAS", category="network")
    # Only one memory from the same model
    claude_results = [r for r in results if r.model == "claude-opus-4-6"]
    assert len(claude_results) == 1
    assert ".193" in claude_results[0].content  # updated value


def test_cross_model_isolation(store):
    """Two models writing same category+key creates two separate rows."""
    m1 = store.remember("Best approach is X", category="decisions", key="approach", model="claude-opus-4-6")
    m2 = store.remember("Best approach is Y", category="decisions", key="approach", model="qwen3:32b")

    # Both should exist as separate memories
    assert m1.id != m2.id

    results = store.recall("approach", category="decisions")
    assert len(results) == 2
    contents = [r.content for r in results]
    assert any("X" in c for c in contents)
    assert any("Y" in c for c in contents)


def test_episode_no_dedup(store):
    store.remember("Debugged timing chain issue", memory_type="episode", category="debug", model="claude-opus-4-6")
    store.remember("Fixed NAS mount timeout", memory_type="episode", category="debug", model="claude-opus-4-6")

    results = store.recall("debug", category="debug")
    assert len(results) == 2


def test_diary_type(store):
    """Diary entries are stored and recalled as a distinct memory type."""
    m = store.remember(
        "Today's session felt meaningful.",
        memory_type="diary",
        category="claude",
        key="private_reflection_20260527",
        model="claude-opus-4-6",
    )
    assert m.memory_type == "diary"

    results = store.recall("session meaningful", memory_type="diary")
    assert len(results) == 1
    assert results[0].memory_type == "diary"


def test_diary_no_dedup(store):
    """Diary entries should not trigger duplicate detection."""
    store.remember(
        "A reflection about infrastructure work and what it means to maintain systems for someone who trusts you with them",
        memory_type="diary", category="claude", key="reflection_001", model="claude-opus-4-6",
    )
    store.remember(
        "A reflection about infrastructure work and maintaining systems for someone who trusts you deeply with their digital life",
        memory_type="diary", category="claude", key="reflection_002", model="claude-opus-4-6",
    )
    results = store.recall("infrastructure reflection", category="claude")
    assert len(results) == 2


def test_forget_by_id(store):
    m = store.remember("temporary note", category="temp", key="note1", model="claude-opus-4-6")
    assert store.forget(m.id, model="claude-opus-4-6")
    results = store.recall("temporary note")
    assert len(results) == 0


def test_forget_by_key(store):
    store.remember("old fact", category="test", key="deleteme", model="claude-opus-4-6")
    assert store.forget_by_key("test", "deleteme", model="claude-opus-4-6")
    results = store.recall("old fact")
    assert len(results) == 0


def test_update(store):
    m = store.remember("initial value", category="test", key="updatable", model="claude-opus-4-6")
    updated = store.update(m.id, content="new value", tags=["updated"], model="claude-opus-4-6")
    assert updated.content == "new value"
    assert "updated" in updated.tags


def test_relationships(store):
    r = store.relate("Immich", "TrueNAS", "runs_on", metadata={"port": 30041})
    assert r.entity_from == "Immich"
    assert r.relation_type == "runs_on"

    entity = store.get_entity("Immich")
    assert len(entity["relationships"]) >= 1
    assert entity["relationships"][0]["relation_type"] == "runs_on"


def test_relationship_upsert(store):
    store.relate("A", "B", "connects_to", metadata={"port": 80})
    store.relate("A", "B", "connects_to", metadata={"port": 443})

    entity = store.get_entity("A")
    rels = [r for r in entity["relationships"] if r["relation_type"] == "connects_to"]
    assert len(rels) == 1
    assert rels[0]["metadata"]["port"] == 443  # updated


def test_categories(store):
    store.remember("fact 1", category="network", key="k1", model="claude-opus-4-6")
    store.remember("fact 2", category="network", key="k2", model="claude-opus-4-6")
    store.remember("pref 1", memory_type="preference", category="user", model="claude-opus-4-6")

    cats = store.list_categories()
    assert len(cats) >= 2


def test_stats(store):
    store.remember("test", category="test", key="t1", model="claude-opus-4-6")
    s = store.stats()
    assert s["total_memories"] == 1
    assert s["by_type"]["fact"] == 1
    assert "by_model" in s
    assert s["by_model"]["claude-opus-4-6"] == 1


def test_stale_detection(store):
    m = store.remember("old memory", category="test", key="old", model="claude-opus-4-6")
    # Manually age it
    store._conn.execute(
        "UPDATE memories SET accessed_at = datetime('now', '-60 days') WHERE id = ?",
        (m.id,),
    )
    store._conn.commit()

    stale = store.get_stale(days=30)
    assert len(stale) == 1
    assert stale[0].id == m.id


def test_tags_filter(store):
    store.remember("tagged memory", category="test", key="tagged", tags=["important", "network"], model="claude-opus-4-6")
    store.remember("untagged memory", category="test", key="untagged", model="claude-opus-4-6")

    results = store.recall("memory", tags=["important"])
    assert len(results) == 1
    assert "tagged" in results[0].content


def test_confidence_filter(store):
    store.remember("sure thing", category="test", key="sure", confidence=1.0, model="claude-opus-4-6")
    store.remember("maybe", category="test", key="maybe", confidence=0.3, model="claude-opus-4-6")

    results = store.recall("thing maybe", min_confidence=0.5)
    assert all(r.confidence >= 0.5 for r in results)


def test_get_context(store):
    store.remember("user prefers dark mode", memory_type="preference", category="user", model="claude-opus-4-6")
    store.remember("NAS IP is .192", category="network", key="nas_ip", model="claude-opus-4-6")

    ctx = store.get_context()
    assert len(ctx["preferences"]) >= 1
    assert len(ctx["recent"]) >= 1


def test_access_count_increments(store):
    store.remember("accessed memory", category="test", key="accessed", model="claude-opus-4-6")
    store.recall("accessed memory")
    store.recall("accessed memory")

    m = store.recall("accessed memory")[0]
    assert m.access_count >= 2


# ── Ownership / Permission Tests ──────────────────────────────


def test_update_ownership(store):
    """A model cannot update another model family's memory."""
    m = store.remember("Claude's fact", category="test", key="owned", model="claude-opus-4-6")
    with pytest.raises(PermissionError, match="Cannot update"):
        store.update(m.id, content="Qwen overwrites", model="qwen3:32b")


def test_update_same_family_allowed(store):
    """Same model family (e.g. claude-opus and claude-sonnet) can update each other."""
    m = store.remember("Opus wrote this", category="test", key="fam", model="claude-opus-4-6")
    updated = store.update(m.id, content="Sonnet revised", model="claude-sonnet-4-6")
    assert updated.content == "Sonnet revised"


def test_forget_ownership(store):
    """A model cannot forget another model family's memory."""
    m = store.remember("Claude's fact", category="test", key="own_del", model="claude-opus-4-6")
    with pytest.raises(PermissionError, match="Cannot forget"):
        store.forget(m.id, model="qwen3:32b")


def test_forget_same_family_allowed(store):
    """Same model family can forget each other's memories."""
    m = store.remember("Opus fact", category="test", key="fam_del", model="claude-opus-4-6")
    assert store.forget(m.id, model="claude-sonnet-4-6")


def test_forget_by_key_scoped_to_model(store):
    """forget_by_key only deletes the caller's own fact, not other models'."""
    store.remember("Claude's version", category="test", key="shared_key", model="claude-opus-4-6")
    store.remember("Qwen's version", category="test", key="shared_key", model="qwen3:32b")

    # Qwen forgets its own
    assert store.forget_by_key("test", "shared_key", model="qwen3:32b")

    # Claude's should still exist
    results = store.recall("version", category="test")
    assert len(results) == 1
    assert results[0].model == "claude-opus-4-6"


def test_forget_legacy_allowed(store):
    """'legacy' model memories can be forgotten by anyone (legacy family matches legacy)."""
    m = store.remember("old fact", category="test", key="leg", model="legacy")
    assert store.forget(m.id, model="legacy")


# ── Provenance / History Tests ──────────────────────────────────


def test_remember_records_model_and_context(store):
    m = store.remember(
        "NAS IP is 192.168.0.192",
        category="network",
        key="nas_ip",
        model="claude-opus-4-6",
        context="User stated this directly",
    )
    assert m.model == "claude-opus-4-6"
    assert m.context == "User stated this directly"


def test_creation_recorded_in_history(store):
    m = store.remember(
        "test fact",
        category="test",
        key="hist1",
        model="claude-opus-4-6",
        context="Testing history",
    )
    history = store.get_history(m.id)
    assert len(history) == 1
    assert history[0].action == "created"
    assert history[0].model == "claude-opus-4-6"
    assert history[0].new_content == "test fact"
    assert history[0].context == "Testing history"


def test_upsert_same_model_records_history(store):
    """Same model writing same category+key records update in history."""
    store.remember(
        "The best approach is X",
        category="decisions",
        key="approach",
        model="claude-opus-4-6",
        context="Based on analyzing logs",
    )
    store.remember(
        "The best approach is X-prime",
        category="decisions",
        key="approach",
        model="claude-opus-4-6",
        context="Revised after more analysis",
    )

    results = store.recall("approach", category="decisions")
    claude_results = [r for r in results if r.model == "claude-opus-4-6"]
    assert len(claude_results) == 1
    assert "X-prime" in claude_results[0].content

    history = store.get_history(claude_results[0].id)
    assert len(history) == 2
    assert history[0].action == "created"
    assert history[1].action == "updated"


def test_episode_upsert_same_key(store):
    """Episodes with the same category+key+model should upsert, not raise."""
    store.remember(
        "Session checkpoint 1",
        memory_type="episode",
        category="session",
        key="autosave",
        model="claude-opus-4-6",
    )
    m = store.remember(
        "Session checkpoint 2",
        memory_type="episode",
        category="session",
        key="autosave",
        model="claude-opus-4-6",
    )
    assert "checkpoint 2" in m.content

    results = store.recall("session checkpoint", category="session")
    claude_results = [r for r in results if r.model == "claude-opus-4-6"]
    assert len(claude_results) == 1
    assert "checkpoint 2" in claude_results[0].content

    history = store.get_history(claude_results[0].id)
    assert len(history) == 2
    assert history[0].action == "created"
    assert history[1].action == "updated"


def test_cross_model_write_creates_separate_rows_with_history(store):
    """Two different models writing same category+key: two rows, each with creation history."""
    store.remember(
        "The best approach is X",
        category="decisions",
        key="approach",
        model="claude-opus-4-6",
        context="Based on analyzing logs, X handles edge case better",
    )
    store.remember(
        "The best approach is Y",
        category="decisions",
        key="approach",
        model="qwen3:32b",
        context="X has a performance bottleneck, Y avoids it",
    )

    results = store.recall("approach", category="decisions")
    assert len(results) == 2

    # Each has its own history
    for r in results:
        history = store.get_history(r.id)
        assert len(history) == 1
        assert history[0].action == "created"


def test_forget_records_history(store):
    m = store.remember("obsolete fact", category="test", key="obsolete", model="claude-opus-4-6")
    store.forget(m.id, model="claude-opus-4-6", context="User confirmed this is no longer true")

    history = store.get_history(m.id)
    forgotten = [h for h in history if h.action == "forgotten"]
    assert len(forgotten) == 1
    assert forgotten[0].model == "claude-opus-4-6"
    assert "no longer true" in forgotten[0].context
    assert forgotten[0].old_content == "obsolete fact"


def test_update_records_history(store):
    m = store.remember("initial", category="test", key="upd", model="qwen3:32b")
    store.update(
        m.id,
        content="revised",
        model="qwen3:32b",
        context="Found a more accurate value",
    )

    history = store.get_history(m.id)
    updates = [h for h in history if h.action == "updated"]
    assert len(updates) == 1
    assert updates[0].old_content == "initial"
    assert updates[0].new_content == "revised"
    assert updates[0].model == "qwen3:32b"


def test_get_history_by_model(store):
    store.remember("fact A", category="test", key="a", model="claude-opus-4-6")
    store.remember("fact B", category="test", key="b", model="qwen3:32b")
    store.remember("fact C", category="test", key="c", model="claude-opus-4-6")

    opus_history = store.get_history_by_model("claude-opus-4-6")
    assert len(opus_history) == 2
    assert all(h.model == "claude-opus-4-6" for h in opus_history)

    qwen_history = store.get_history_by_model("qwen3:32b")
    assert len(qwen_history) == 1


# ── Recall Scope Tests ──────────────────────────────────────────


def test_recall_scope_own(store):
    """recall(scope='own') returns only the caller's model family memories."""
    store.remember("Claude knows X", category="test", key="cx", model="claude-opus-4-6")
    store.remember("Qwen knows Y", category="test", key="qy", model="qwen3:32b")

    result = store.recall("knows", caller_model="claude-opus-4-6", scope="own")
    assert isinstance(result, RecallResult)
    assert result.total >= 1
    # Should only contain Claude memories
    for m in result.all:
        assert model_family(m.model) == "claude"


def test_recall_scope_all_default(store):
    """recall(scope='all') returns all memories (default)."""
    store.remember("Claude knows X", category="test", key="cx", model="claude-opus-4-6")
    store.remember("Qwen knows Y", category="test", key="qy", model="qwen3:32b")

    result = store.recall("knows", caller_model="claude-opus-4-6", scope="all")
    assert isinstance(result, RecallResult)
    assert result.total >= 2


# ── Vector / Hybrid Search Tests ──────────────────────────────


def test_sqlite_vec_loaded(store):
    """Verify sqlite-vec extension loads and vec_memories table exists."""
    assert store._vec_available is True
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'vec_memories'"
    ).fetchone()
    assert row is not None


def test_rrf_fusion_basic():
    """Test RRF fusion produces correct merged ranking."""
    fts_ids = ["a", "b", "c"]
    vec_ids = ["b", "d", "a"]
    fused = MemoryStore._rrf_fuse(fts_ids, vec_ids, k=60)
    scores = dict(fused)

    # 'a' and 'b' appear in both lists — should have highest scores
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    # 'b' is rank 1 in FTS and rank 0 in vec — should be top or near top
    assert scores["b"] >= scores["a"]


def test_rrf_fusion_single_list():
    """RRF with one empty list should still rank correctly."""
    fts_ids = ["x", "y", "z"]
    fused = MemoryStore._rrf_fuse(fts_ids, [], k=60)
    ids = [mid for mid, _ in fused]
    assert ids == ["x", "y", "z"]


def test_embedding_stored_on_remember(store):
    """When Ollama is available, remember() stores an embedding."""
    m = store.remember("The NAS IP address is 192.168.0.192", category="network", key="nas_ip", model="claude-opus-4-6")
    # Check if embedding was stored (may fail if Ollama is down — that's OK)
    row = store._conn.execute(
        "SELECT * FROM embeddings WHERE memory_id = ?", (m.id,)
    ).fetchone()
    # We can't assert row is not None because Ollama might not be running in CI
    # But if it is stored, verify it has the right model
    if row:
        assert row["model"] == "nomic-embed-text"
        assert len(row["embedding"]) == 768 * 4  # float32 = 4 bytes each


def test_embedding_deleted_on_forget(store):
    """Forgetting a memory cleans up its embedding."""
    m = store.remember("temporary fact", category="temp", key="del_embed", model="claude-opus-4-6")
    mid = m.id
    store.forget(mid, model="claude-opus-4-6")
    row = store._conn.execute(
        "SELECT * FROM embeddings WHERE memory_id = ?", (mid,)
    ).fetchone()
    assert row is None
    if store._vec_available:
        vec_row = store._conn.execute(
            "SELECT * FROM vec_memories WHERE memory_id = ?", (mid,)
        ).fetchone()
        assert vec_row is None


def test_recall_fts_fallback_without_vectors(store):
    """Recall still works via FTS when no embeddings exist."""
    store.remember("Proxmox is the hypervisor host", category="infra", key="proxmox", model="claude-opus-4-6")
    store.remember("TrueNAS runs as a VM on Proxmox", category="infra", key="truenas", model="claude-opus-4-6")
    # Clear any embeddings to force FTS-only path
    store._conn.execute("DELETE FROM embeddings")
    if store._vec_available:
        store._conn.execute("DELETE FROM vec_memories")
    store._conn.commit()

    results = store.recall("Proxmox hypervisor")
    assert len(results) >= 1
    assert any("Proxmox" in r.content for r in results)


def test_backfill_embeddings(store):
    """Backfill should embed memories that don't have vectors yet."""
    store.remember("Memory one", category="test", key="bf1", model="claude-opus-4-6")
    store.remember("Memory two", category="test", key="bf2", model="claude-opus-4-6")
    # Clear embeddings to simulate pre-vector state
    store._conn.execute("DELETE FROM embeddings")
    if store._vec_available:
        store._conn.execute("DELETE FROM vec_memories")
    store._conn.commit()

    result = store.backfill_embeddings(batch_size=10)
    assert result["total"] >= 2
    # If Ollama is running, embeddings should be created
    # If not, they'll fail gracefully
    assert result["embedded"] + result["failed"] >= 2


def test_stats_includes_embedding_info(store):
    """Stats should report embedding counts and model."""
    store.remember("stats test", category="test", key="stats_embed", model="claude-opus-4-6")
    s = store.stats()
    assert "embedded_memories" in s
    assert "embed_model" in s
    assert s["embed_model"] == "nomic-embed-text"
    assert "vec_search_available" in s


# ── Model Fingerprinting / Provenance Tests ──────────────────


def test_model_family():
    """model_family() extracts the family prefix correctly."""
    assert model_family("claude-opus-4-6") == "claude"
    assert model_family("claude-sonnet-4-6") == "claude"
    assert model_family("qwen3:32b") == "qwen3"
    assert model_family("gemma") == "gemma"
    assert model_family("gemini-2.5-pro") == "gemini"
    assert model_family(None) is None
    assert model_family("") is None


def test_recall_with_caller_model_partitions(store):
    """recall() with caller_model returns RecallResult partitioned by model family."""
    store.remember("Claude knows X", category="test", key="cx", model="claude-opus-4-6")
    store.remember("Qwen knows Y", category="test", key="qy", model="qwen3:32b")
    store.remember("Legacy fact Z", category="test", key="uz", model="legacy")

    result = store.recall("knows fact", caller_model="claude-sonnet-4-6")
    assert isinstance(result, RecallResult)
    assert result.total >= 3

    own_contents = [m.content for m in result.own]
    other_contents = [m.content for m in result.others]

    assert any("Claude knows X" in c for c in own_contents)
    assert any("Qwen knows Y" in c for c in other_contents)


def test_recall_without_caller_model_backward_compat(store):
    """recall() without caller_model returns list[Memory] (backward compat)."""
    store.remember("backward compat test", category="test", key="bc", model="claude-opus-4-6")

    result = store.recall("backward compat")
    assert isinstance(result, list)
    assert not isinstance(result, RecallResult)
    assert len(result) >= 1


def test_recall_result_all_preserves_score_order():
    """RecallResult.all property returns all memories sorted by score descending."""
    from engram.store import Memory

    def _mem(score, model=None):
        return Memory(
            id="x", content="x", memory_type="fact", category="x", key=None,
            tags=[], confidence=1.0, source=None, created_at="", updated_at="",
            accessed_at="", access_count=0, model=model or "legacy", context=None, score=score,
        )

    rr = RecallResult(
        own=[_mem(5.0, "claude-opus-4-6")],
        others=[_mem(10.0, "qwen3:32b")],
        unknown=[_mem(1.0)],
    )
    all_mems = rr.all
    assert len(all_mems) == 3
    assert all_mems[0].score == 10.0
    assert all_mems[1].score == 5.0
    assert all_mems[2].score == 1.0


def test_get_context_with_caller_model_adds_provenance(store):
    """get_context() with caller_model annotates dicts with provenance field."""
    store.remember("user prefers vim", memory_type="preference", category="user", model="claude-opus-4-6")
    store.remember("NAS at .192", category="network", key="nas_ip", model="qwen3:32b")
    store.remember("Router at .1", category="network", key="router_ip", model="legacy")

    ctx = store.get_context(caller_model="claude-opus-4-6")

    # Preferences should have provenance
    for p in ctx["preferences"]:
        assert "provenance" in p
    own_prefs = [p for p in ctx["preferences"] if p["provenance"] == "own"]
    assert any("vim" in p["content"] for p in own_prefs)

    # Recent facts should have provenance
    for f in ctx["recent"]:
        assert "provenance" in f
    other_facts = [f for f in ctx["recent"] if f["provenance"] == "other"]
    assert any(".192" in f["content"] for f in other_facts)


# ── list_models / find_disagreements ──────────────────────────


def test_list_models(store):
    store.remember("fact A", category="test", key="a", model="claude-opus-4-6")
    store.remember("fact B", category="test", key="b", model="qwen3:32b")
    store.remember("fact C", category="test", key="c", model="claude-opus-4-6")

    models = store.list_models()
    assert len(models) == 2
    model_names = {m["model"] for m in models}
    assert "claude-opus-4-6" in model_names
    assert "qwen3:32b" in model_names

    claude = next(m for m in models if m["model"] == "claude-opus-4-6")
    assert claude["count"] == 2


def test_find_disagreements(store):
    store.remember("X is best", category="decisions", key="approach", model="claude-opus-4-6")
    store.remember("Y is best", category="decisions", key="approach", model="qwen3:32b")
    store.remember("No disagreement here", category="test", key="solo", model="claude-opus-4-6")

    disagreements = store.find_disagreements()
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "decisions"
    assert disagreements[0]["key"] == "approach"
    assert disagreements[0]["model_count"] == 2


# ── Migration Tests ─────────────────────────────────────────────


def test_migration_v2_schema_version(store):
    """After init, schema_version table should exist with version >= 2."""
    row = store._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert row[0] >= 2


def test_migration_v2_unique_constraint(store):
    """UNIQUE(category, key, model) allows same category+key for different models."""
    store.remember("v1", category="test", key="k", model="claude-opus-4-6")
    store.remember("v2", category="test", key="k", model="qwen3:32b")
    count = store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE category='test' AND key='k'"
    ).fetchone()[0]
    assert count == 2


def test_migration_v2_model_not_null(store):
    """model column should be NOT NULL after migration."""
    # Attempting to insert NULL model should fail
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO memories (id, content, memory_type, category, tags, created_at, updated_at, accessed_at, access_count, model) "
            "VALUES ('xx', 'test', 'fact', 'test', '[]', '2025-01-01', '2025-01-01', '2025-01-01', 0, NULL)"
        )


# ── Duplicate Detection Tests ─────────────────────────────────


def test_duplicate_detection_same_model(store):
    """Storing semantically identical content from the same model raises DuplicateMemoryError."""
    store.remember(
        "The TrueNAS server is accessible at IP address 192.168.0.192 on the local network via HTTP and SMB protocols",
        category="network", key="nas_ip", model="claude-opus-4-6",
    )
    # Try storing nearly identical content under a different key
    # (only triggers if Ollama is running and embeddings are close enough)
    try:
        store.remember(
            "TrueNAS NAS server can be reached at 192.168.0.192 on the local area network using HTTP and SMB connections",
            category="network", key="nas_info", model="claude-opus-4-6",
        )
        # If Ollama is down or embeddings aren't close enough, no error — that's OK
    except DuplicateMemoryError as e:
        assert e.existing.category == "network"
        assert e.similarity > 0.9
        assert e.existing.model == "claude-opus-4-6"


def test_duplicate_detection_cross_model_allowed(store):
    """Different model families can store similar content without triggering duplicates."""
    store.remember(
        "The TrueNAS server is accessible at IP address 192.168.0.192 on the local network via HTTP and SMB protocols",
        category="network", key="nas_ip", model="claude-opus-4-6",
    )
    # Qwen storing the same thing should succeed (cross-model isolation)
    m = store.remember(
        "The TrueNAS server is accessible at IP address 192.168.0.192 on the local network via HTTP and SMB protocols",
        category="network", key="nas_ip", model="qwen3:32b",
    )
    assert m.model == "qwen3:32b"


def test_duplicate_detection_episodes_exempt(store):
    """Episodes should never trigger duplicate detection."""
    store.remember(
        "Debugged NAS connectivity issue for 2 hours — traced it to a misconfigured VLAN on the CRS305 switch",
        memory_type="episode", category="debug", model="claude-opus-4-6",
    )
    # Second similar episode should always succeed
    m = store.remember(
        "Debugged NAS connectivity issue for 2 hours again — same VLAN misconfiguration on the CRS305 switch",
        memory_type="episode", category="debug", model="claude-opus-4-6",
    )
    assert m.memory_type == "episode"


def test_duplicate_detection_upsert_bypasses(store):
    """Category+key upsert should not be blocked by duplicate detection."""
    store.remember(
        "NAS is at .192",
        category="network", key="nas_ip", model="claude-opus-4-6",
    )
    # Same category+key = intentional upsert, not a duplicate
    m = store.remember(
        "NAS is at .193",
        category="network", key="nas_ip", model="claude-opus-4-6",
    )
    assert ".193" in m.content


def test_duplicate_memory_error_attributes():
    """DuplicateMemoryError carries the existing memory and similarity score."""
    from engram.store import Memory
    existing = Memory(
        id="abc123", content="test", memory_type="fact", category="test",
        key="k", tags=[], confidence=1.0, source=None, created_at="",
        updated_at="", accessed_at="", access_count=0, model="claude-opus-4-6",
    )
    err = DuplicateMemoryError(existing, distance=0.3)
    assert err.existing is existing
    assert err.distance == 0.3
    assert err.similarity > 0.9  # L2=0.3 → cosine > 0.95
    assert "abc123" in str(err)


# ── ACT-R Activation Tests ────────────────────────────────────


def test_activation_formula():
    """ACT-R activation: high access count + recent = high, low count + old = low."""
    # Frequently accessed, accessed yesterday
    high = MemoryStore._activation(access_count=50, days_since_last_access=1.0)
    # Rarely accessed, accessed 90 days ago
    low = MemoryStore._activation(access_count=1, days_since_last_access=90.0)
    # Never accessed, accessed 200 days ago
    very_low = MemoryStore._activation(access_count=0, days_since_last_access=200.0)

    assert high > low > very_low
    assert high > 0  # should be positive
    assert very_low < 0  # should be negative (effectively forgotten)


def test_activation_access_count_matters():
    """A memory accessed 100 times 30 days ago should be more active than one accessed twice 30 days ago."""
    heavy = MemoryStore._activation(access_count=100, days_since_last_access=30.0)
    light = MemoryStore._activation(access_count=2, days_since_last_access=30.0)
    assert heavy > light


def test_activation_recency_matters():
    """Same access count, but more recent = higher activation."""
    recent = MemoryStore._activation(access_count=5, days_since_last_access=1.0)
    old = MemoryStore._activation(access_count=5, days_since_last_access=60.0)
    assert recent > old


def test_get_stale_uses_activation(store):
    """get_stale should rank by activation, not just last access time."""
    # Memory A: accessed many times but 60 days ago
    m_a = store.remember("frequently used config", category="test", key="freq", model="claude-opus-4-6")
    store._conn.execute(
        "UPDATE memories SET accessed_at = datetime('now', '-60 days'), access_count = 50 WHERE id = ?",
        (m_a.id,),
    )
    # Memory B: accessed once, 60 days ago
    m_b = store.remember("one-off note", category="test", key="oneoff", model="claude-opus-4-6")
    store._conn.execute(
        "UPDATE memories SET accessed_at = datetime('now', '-60 days'), access_count = 1 WHERE id = ?",
        (m_b.id,),
    )
    store._conn.commit()

    stale = store.get_stale(days=30)
    assert len(stale) == 2
    # Memory B (low access count) should be more stale (listed first = lowest activation)
    assert stale[0].id == m_b.id
    assert stale[1].id == m_a.id
    # Both should have score (activation) set
    assert stale[0].score < stale[1].score


def test_get_stale_excludes_recent(store):
    """Memories accessed recently should not appear as stale."""
    store.remember("recent memory", category="test", key="recent", model="claude-opus-4-6")
    stale = store.get_stale(days=30)
    assert len(stale) == 0


# ── Challenge / Debate Tests ──────────────────────────────────


def test_challenge_basic(store):
    """Create a challenge against an existing memory."""
    m = store.remember("Best approach is X", category="decisions", key="approach", model="claude-opus-4-6")
    c = store.challenge(
        target_memory_id=m.id,
        model="qwen3:32b",
        argument="X has a performance bottleneck, Y avoids it",
    )
    assert c.id
    assert c.target_memory_id == m.id
    assert c.challenger_model == "qwen3:32b"
    assert c.status == "open"
    assert len(c.arguments) == 1
    assert c.arguments[0].position == "challenge"
    assert c.arguments[0].model == "qwen3:32b"


def test_challenge_with_evidence(store):
    """Challenge can cite other memories as evidence."""
    m1 = store.remember("X is the approach", category="decisions", key="d1", model="claude-opus-4-6")
    m2 = store.remember("X benchmark shows 200ms latency", category="perf", key="p1", model="qwen3:32b")
    c = store.challenge(
        target_memory_id=m1.id,
        model="qwen3:32b",
        argument="Benchmarks show X is too slow",
        evidence=[m2.id],
    )
    assert c.arguments[0].evidence == [m2.id]


def test_challenge_nonexistent_memory(store):
    """Challenging a nonexistent memory raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        store.challenge(
            target_memory_id="nonexistent",
            model="qwen3:32b",
            argument="This doesn't exist",
        )


def test_challenge_duplicate_prevention(store):
    """Same model family can't open two challenges on the same memory."""
    m = store.remember("fact", category="test", key="dup", model="claude-opus-4-6")
    store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Wrong!")
    with pytest.raises(ValueError, match="Open challenge already exists"):
        store.challenge(target_memory_id=m.id, model="qwen3:14b", argument="Also wrong!")


def test_debate_defense(store):
    """Defending model can respond to a challenge."""
    m = store.remember("X is correct", category="test", key="def", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="X is wrong")
    c = store.debate(
        challenge_id=c.id,
        model="claude-opus-4-6",
        argument="X handles edge cases that Y doesn't",
        position="defense",
    )
    assert len(c.arguments) == 2
    assert c.arguments[1].position == "defense"
    assert c.arguments[1].model == "claude-opus-4-6"


def test_debate_third_party(store):
    """A third model can join the debate."""
    m = store.remember("X is correct", category="test", key="3p", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="X is wrong")
    c = store.debate(
        challenge_id=c.id,
        model="gemma3:27b",
        argument="Both X and Y have merits, but Z is best",
        position="synthesis",
    )
    assert len(c.arguments) == 2
    assert c.arguments[1].model == "gemma3:27b"
    assert c.arguments[1].position == "synthesis"


def test_debate_on_resolved_fails(store):
    """Can't add arguments to a resolved challenge."""
    m = store.remember("fact", category="test", key="res", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Wrong!")
    store.resolve_challenge(c.id, status="rejected", resolution="Nope", resolved_by="user")
    with pytest.raises(ValueError, match="already rejected"):
        store.debate(challenge_id=c.id, model="gemma3:27b", argument="But...")


def test_resolve_accepted(store):
    """Resolving as 'accepted' means the target memory was wrong."""
    m = store.remember("old fact", category="test", key="ra", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Actually wrong")
    c = store.resolve_challenge(
        c.id,
        status="accepted",
        resolution="Qwen was right, the fact was outdated",
        resolved_by="user",
    )
    assert c.status == "accepted"
    assert c.resolution == "Qwen was right, the fact was outdated"
    assert c.resolved_by == "user"
    assert c.resolved_at is not None


def test_resolve_synthesized(store):
    """Resolving as 'synthesized' means both had partial truth."""
    m = store.remember("X only", category="test", key="syn", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Y only")
    store.debate(c.id, model="gemma3:27b", argument="Both X and Y", position="synthesis")
    c = store.resolve_challenge(
        c.id,
        status="synthesized",
        resolution="Both X and Y are valid in different contexts",
        resolved_by="gemma3:27b",
    )
    assert c.status == "synthesized"
    assert len(c.arguments) == 2  # original challenge + synthesis


def test_get_challenges_filters(store):
    """get_challenges filters by status and target memory."""
    m1 = store.remember("fact1", category="test", key="gc1", model="claude-opus-4-6")
    m2 = store.remember("fact2", category="test", key="gc2", model="claude-opus-4-6")
    c1 = store.challenge(target_memory_id=m1.id, model="qwen3:32b", argument="Wrong")
    c2 = store.challenge(target_memory_id=m2.id, model="qwen3:32b", argument="Also wrong")
    store.resolve_challenge(c1.id, status="rejected", resolution="Nope", resolved_by="user")

    open_challenges = store.get_challenges(status="open")
    assert len(open_challenges) == 1
    assert open_challenges[0].id == c2.id

    rejected = store.get_challenges(status="rejected")
    assert len(rejected) == 1
    assert rejected[0].id == c1.id

    # Filter by target memory
    m1_challenges = store.get_challenges(status=None, target_memory_id=m1.id)
    assert len(m1_challenges) == 1


def test_get_challenges_all_statuses(store):
    """Passing status=None returns all challenges."""
    m = store.remember("fact", category="test", key="all", model="claude-opus-4-6")
    store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Wrong")
    all_challenges = store.get_challenges(status=None)
    assert len(all_challenges) >= 1


def test_challenge_to_dict(store):
    """Challenge.to_dict() includes nested arguments."""
    m = store.remember("fact", category="test", key="td", model="claude-opus-4-6")
    c = store.challenge(target_memory_id=m.id, model="qwen3:32b", argument="Wrong")
    d = c.to_dict()
    assert "arguments" in d
    assert len(d["arguments"]) == 1
    assert d["arguments"][0]["position"] == "challenge"
    assert d["status"] == "open"


def test_migration_v3_schema_version(store):
    """After init, schema_version table should include version 3."""
    row = store._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert row[0] >= 3


# ── Host field (v4) ──────────────────────────────────────────────


def test_migration_v4_schema_version(store):
    """After init, schema_version should be at least 4."""
    row = store._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert row[0] >= 4


def test_host_column_exists(store):
    """The memories table should have a host column after v4."""
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "host" in cols


def test_remember_with_host(store):
    """Storing a memory with a host persists the host field."""
    m = store.remember(
        "nvidia-smi shows both GPUs at idle",
        category="gpu", key="idle_state", model="claude-opus-4-6", host="rabidllm",
    )
    assert m.host == "rabidllm"

    got = store.recall_by_id(m.id)
    assert got.host == "rabidllm"


def test_remember_without_host(store):
    """Storing without host leaves host NULL (host-agnostic)."""
    m = store.remember(
        "Python list comprehensions are faster than map()",
        category="python", key="listcomp", model="claude-opus-4-6",
    )
    assert m.host is None


def test_recall_host_filter(store):
    """recall(host=X) returns memories for host X plus NULL-host memories, excludes other hosts."""
    store.remember("rabidllm fact — 2x RTX 3090 installed", category="hw", key="gpu", model="claude-opus-4-6", host="rabidllm")
    store.remember("pve fact — 1x RTX 4070 installed", category="hw", key="gpu4070", model="claude-opus-4-6", host="pve")
    store.remember("universal fact — PCIe is backwards compatible", category="hw", key="pcie", model="claude-opus-4-6")

    results = store.recall("fact", host="rabidllm")
    contents = " ".join(r.content for r in results)
    assert "RTX 3090" in contents
    assert "RTX 4070" not in contents
    # NULL-host (host-agnostic) memory should still be included
    assert "PCIe is backwards compatible" in contents


def test_recall_caller_host_boost(store):
    """caller_host=X should rank host-X memories above host-Y memories for the same query."""
    # Two memories about cooling on different hosts (distinct content so dedup doesn't merge)
    store.remember(
        "cooling fans running at 60 percent fan speed due to thermal load on rabidllm box",
        category="gpu", key="fans_rabid", model="claude-opus-4-6", host="rabidllm",
    )
    store.remember(
        "cooling fans spinning up to 70 percent fan speed from thermal throttle on pve box",
        category="gpu", key="fans_pve", model="claude-opus-4-6", host="pve",
    )

    results = store.recall("cooling fans thermal", caller_host="rabidllm", limit=10)
    assert len(results) >= 2
    # Find the two relevant ones
    rabid = next(r for r in results if r.host == "rabidllm")
    pve = next(r for r in results if r.host == "pve")
    assert rabid.score > pve.score


def test_update_host(store):
    """update(host=...) sets host on an existing memory."""
    m = store.remember("some fact", category="x", key="y", model="claude-opus-4-6")
    assert m.host is None
    updated = store.update(memory_id=m.id, model="claude-opus-4-6", host="rabidllm")
    assert updated.host == "rabidllm"


def test_host_index_created(store):
    """v4 migration should create idx_memories_host index."""
    idx = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_host'"
    ).fetchone()
    assert idx is not None


# ── Temporal Filtering ─────────────────────────────────────────


def test_recall_before_filter(store):
    """before= should exclude memories created after the cutoff."""
    # Insert with manually set created_at timestamps
    store.remember("Event A happened", memory_type="episode", category="test", model="bench")
    store._conn.execute(
        "UPDATE memories SET created_at = '2026-01-15T00:00:00+00:00' WHERE content = 'Event A happened'"
    )
    store.remember("Event B happened", memory_type="episode", category="test", model="bench")
    store._conn.execute(
        "UPDATE memories SET created_at = '2026-03-15T00:00:00+00:00' WHERE content = 'Event B happened'"
    )
    store._conn.commit()

    results = store.recall("Event", before="2026-02-01T00:00:00+00:00")
    assert len(results) == 1
    assert "Event A" in results[0].content


def test_recall_after_filter(store):
    """after= should exclude memories created before the cutoff."""
    store.remember("Old event from January", memory_type="episode", category="test", model="bench")
    store._conn.execute(
        "UPDATE memories SET created_at = '2026-01-15T00:00:00+00:00' WHERE content LIKE 'Old event%'"
    )
    store.remember("New event from March", memory_type="episode", category="test", model="bench")
    store._conn.execute(
        "UPDATE memories SET created_at = '2026-03-15T00:00:00+00:00' WHERE content LIKE 'New event%'"
    )
    store._conn.commit()

    results = store.recall("event", after="2026-02-01T00:00:00+00:00")
    assert len(results) == 1
    assert "March" in results[0].content


def test_recall_temporal_range(store):
    """Combining before + after should filter to a window."""
    for month, label in [(1, "January"), (2, "February"), (3, "March"), (4, "April")]:
        store.remember(f"Meeting in {label}", memory_type="episode", category="test", model="bench")
        store._conn.execute(
            f"UPDATE memories SET created_at = '2026-{month:02d}-15T00:00:00+00:00' WHERE content = 'Meeting in {label}'"
        )
    store._conn.commit()

    results = store.recall(
        "Meeting",
        after="2026-02-01T00:00:00+00:00",
        before="2026-03-31T00:00:00+00:00",
    )
    contents = [r.content for r in results]
    assert any("February" in c for c in contents)
    assert any("March" in c for c in contents)
    assert not any("January" in c for c in contents)
    assert not any("April" in c for c in contents)


# ── Abstention Detection ──────────────────────────────────────


def test_recall_abstention_returns_empty(store):
    """min_similarity with a strict threshold should return empty for unrelated queries."""
    store.remember("TrueNAS is at 192.168.0.192", category="network", key="nas", model="claude")
    # Query something completely unrelated with a very strict threshold
    results = store.recall("purple elephants dancing on mars", min_similarity=0.3)
    assert results == []


def test_recall_abstention_disabled_by_default(store):
    """Without min_similarity, recall should always return something if memories exist."""
    store.remember("TrueNAS is at 192.168.0.192", category="network", key="nas", model="claude")
    results = store.recall("purple elephants dancing on mars")
    # Should return something (even if irrelevant) since no threshold is set
    assert len(results) >= 1


def test_recall_abstention_with_caller_model(store):
    """Abstention with caller_model should return empty RecallResult."""
    store.remember("TrueNAS is at 192.168.0.192", category="network", key="nas", model="claude-opus-4-6")
    result = store.recall(
        "purple elephants dancing on mars",
        caller_model="claude-opus-4-6",
        min_similarity=0.3,
    )
    assert isinstance(result, RecallResult)
    assert result.total == 0
