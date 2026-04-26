"""Tests for MemoryIndex — SQLite sidecar index for MEMORY.md."""

import time
from pathlib import Path

import pytest

from nanobot.agent.memory_index import ConsistencyIssue, ItemRecord, MemoryIndex, Relation


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)


class TestSchemaCreation:
    def test_tables_exist(self, index: MemoryIndex):
        tables = index.list_tables()
        assert {"memory_items", "concepts", "item_concepts", "relations",
                "consistency_issues", "metadata"} <= set(tables)

    def test_metadata_defaults(self, index: MemoryIndex):
        assert index.get_meta("schema_version") == "1"
        assert index.get_meta("embedding_dim") == "1536"
        assert index.get_meta("last_reconciled_commit") is None

    def test_set_and_get_meta(self, index: MemoryIndex):
        index.set_meta("last_reconciled_commit", "abc123")
        assert index.get_meta("last_reconciled_commit") == "abc123"


class TestItemsCRUD:
    def _make_item(self, content: str = "foo", section: str = "root") -> ItemRecord:
        now = int(time.time())
        return ItemRecord(
            id="",  # filled by upsert
            source_file="memory/MEMORY.md",
            section_path=section,
            item_type="list_item",
            content=content,
            content_hash=f"hash-{content}",
            embedding=None,
            created_at=now,
            updated_at=now,
            removed_at=None,
        )

    def test_upsert_assigns_id(self, index: MemoryIndex):
        item = self._make_item()
        item_id = index.upsert_item(item)
        assert item_id
        assert index.get_item(item_id).content == "foo"

    def test_list_items_alive_only_by_default(self, index: MemoryIndex):
        a = index.upsert_item(self._make_item("a"))
        b = index.upsert_item(self._make_item("b"))
        index.tombstone_item(b)
        ids = [it.id for it in index.list_items()]
        assert a in ids and b not in ids

    def test_list_items_include_tombstoned(self, index: MemoryIndex):
        a = index.upsert_item(self._make_item("a"))
        index.tombstone_item(a)
        ids = [it.id for it in index.list_items(alive=False)]
        assert a in ids

    def test_upsert_existing_id_updates_content(self, index: MemoryIndex):
        item = self._make_item("v1")
        item_id = index.upsert_item(item)
        item.id = item_id
        item.content = "v2"
        item.content_hash = "hash-v2"
        index.upsert_item(item)
        assert index.get_item(item_id).content == "v2"

    def test_find_similar_items_returns_hits_above_threshold(self, index: MemoryIndex):
        import struct
        def vec(vals): return struct.pack(f"{len(vals)}f", *vals)
        # fixture uses embedding_dim=1536; use 3 floats padded with zeros
        def pad(vals):
            return vec(list(vals) + [0.0] * (1536 - len(vals)))
        a = self._make_item("a"); a.embedding = pad([1.0, 0.0, 0.0])
        b = self._make_item("b"); b.embedding = pad([0.99, 0.1, 0.0])
        c = self._make_item("c"); c.embedding = pad([0.0, 1.0, 0.0])
        index.upsert_item(a); index.upsert_item(b); index.upsert_item(c)
        results = index.find_similar_items(pad([1.0, 0.0, 0.0]), top_k=2, threshold=0.9)
        assert len(results) == 2
        # highest first
        assert results[0][1] >= results[1][1]


from nanobot.agent.memory_index import Concept


class TestConceptsCRUD:
    def _make_concept(self, name: str = "三档待办判断") -> Concept:
        now = int(time.time())
        return Concept(
            id="",
            name=name,
            description=f"desc of {name}",
            centroid_embed=None,
            member_count=0,
            created_at=now,
            updated_at=now,
            merged_into=None,
        )

    def test_upsert_concept_assigns_id(self, index: MemoryIndex):
        c_id = index.upsert_concept(self._make_concept())
        assert c_id and index.get_concept(c_id).name == "三档待办判断"

    def test_find_concept_by_name_fuzzy(self, index: MemoryIndex):
        index.upsert_concept(self._make_concept("三档待办判断"))
        index.upsert_concept(self._make_concept("单聊覆盖"))
        results = index.find_concept_by_name("三档", fuzzy=True)
        assert len(results) == 1
        assert results[0].name == "三档待办判断"

    def test_merge_concept_sets_merged_into(self, index: MemoryIndex):
        a = index.upsert_concept(self._make_concept("A"))
        b = index.upsert_concept(self._make_concept("B"))
        index.merge_concept(a, b)
        assert index.get_concept(a).merged_into == b
        assert index.get_concept(b).merged_into is None

    def test_dedup_concepts_by_name_merges_same_name_into_oldest(self, index: MemoryIndex):
        import time as _t
        # oldest first
        a = index.upsert_concept(self._make_concept("飞书监控群"))
        _t_sleep = _t.sleep  # noqa
        # force distinct created_at via raw update
        b = index.upsert_concept(self._make_concept("飞书监控群"))
        c = index.upsert_concept(self._make_concept("飞书监控群"))
        # Adjust created_at so ordering is stable (a < b < c)
        index._db.execute("UPDATE concepts SET created_at = 1000 WHERE id = ?", (a,))
        index._db.execute("UPDATE concepts SET created_at = 2000 WHERE id = ?", (b,))
        index._db.execute("UPDATE concepts SET created_at = 3000 WHERE id = ?", (c,))
        index._db.commit()

        merged = index.dedup_concepts_by_name()
        assert merged == 2
        # a survives, b and c merged into a
        assert index.get_concept(a).merged_into is None
        assert index.get_concept(b).merged_into == a
        assert index.get_concept(c).merged_into == a


class TestItemConceptLinks:
    def test_link_and_list(self, index: MemoryIndex):
        import time as _t
        item = ItemRecord(
            id="", source_file="memory/MEMORY.md", section_path="r",
            item_type="list_item", content="x", content_hash="h",
            embedding=None, created_at=int(_t.time()), updated_at=int(_t.time()),
            removed_at=None,
        )
        item_id = index.upsert_item(item)
        c_id = index.upsert_concept(Concept(
            id="", name="C", description=None, centroid_embed=None,
            member_count=0, created_at=int(_t.time()), updated_at=int(_t.time()),
            merged_into=None,
        ))
        index.link_item_concept(item_id, c_id, confidence=0.9, source="llm")
        assert index.list_concepts_for_item(item_id) == [c_id]
        assert index.list_items_for_concept(c_id) == [item_id]


class TestRelations:
    def _make_rel(self, from_id: str, to_id: str, rtype: str = "references") -> Relation:
        now = int(time.time())
        return Relation(
            id="", from_kind="item", from_id=from_id, to_kind="item", to_id=to_id,
            relation_type=rtype, confidence=0.8, source="llm",
            rationale=None, created_at=now, invalidated_at=None,
        )

    def test_add_and_query_both_directions(self, index: MemoryIndex):
        r = self._make_rel("A", "B")
        r_id = index.add_relation(r)
        assert r_id
        outgoing = index.relations_from("item", "A")
        incoming = index.relations_to("item", "B")
        assert len(outgoing) == 1 and outgoing[0].to_id == "B"
        assert len(incoming) == 1 and incoming[0].from_id == "A"

    def test_invalidate_relation_filters_from_queries(self, index: MemoryIndex):
        r_id = index.add_relation(self._make_rel("A", "B"))
        index.invalidate_relation(r_id)
        assert index.relations_from("item", "A") == []
        assert index.relations_to("item", "B") == []


class TestImpact:
    def test_query_impact_returns_direct_item_to_item_edge(self, index: MemoryIndex):
        index.add_relation(Relation(
            id="", from_kind="item", from_id="X", to_kind="item", to_id="Y",
            relation_type="depends_on", confidence=0.9, source="llm",
            rationale=None, created_at=int(time.time()), invalidated_at=None,
        ))
        result = index.query_impact("item", "Y")
        assert any(e.from_id == "X" for e in result.incoming_edges)


class TestIssues:
    def test_add_and_list_open(self, index: MemoryIndex):
        now = int(time.time())
        issue = ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed",
            subject_ids='[{"kind":"item","id":"x"}]',
            description="test", severity="medium", status="open",
            resolution=None, created_at=now, resolved_at=None,
        )
        index.add_issue(issue)
        opens = index.list_open_issues()
        assert len(opens) == 1 and opens[0].severity == "medium"

    def test_resolve_issue(self, index: MemoryIndex):
        now = int(time.time())
        issue = ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed",
            subject_ids="[]", description="t", severity="low", status="open",
            resolution=None, created_at=now, resolved_at=None,
        )
        issue_id = index.add_issue(issue)
        index.resolve_issue(issue_id, "ignored", "user ignored")
        assert index.list_open_issues() == []
