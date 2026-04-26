"""Tests for md_chunker — parses MEMORY.md into ordered item chunks."""

from nanobot.agent.md_chunker import Chunk, chunk_markdown


class TestHeadings:
    def test_h2_produces_heading_chunk_with_section_path(self):
        md = "## 飞书消息日报\n"
        chunks = chunk_markdown(md)
        assert len(chunks) == 1
        assert chunks[0].item_type == "heading"
        assert chunks[0].content == "## 飞书消息日报"
        assert chunks[0].section_path == "飞书消息日报"

    def test_h3_under_h2_produces_nested_path(self):
        md = "## 飞书消息日报\n### 触发时间\n"
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert chunks[1].section_path == "飞书消息日报 / 触发时间"


class TestListItems:
    def test_list_items_each_become_chunk(self):
        md = "## A\n- item 1\n- item 2\n"
        chunks = chunk_markdown(md)
        list_chunks = [c for c in chunks if c.item_type == "list_item"]
        assert len(list_chunks) == 2
        assert list_chunks[0].content == "- item 1"
        assert list_chunks[0].section_path == "A"

    def test_nested_list_inherits_parent_indent_as_content(self):
        md = "## A\n- outer\n  - inner\n"
        chunks = chunk_markdown(md)
        list_chunks = [c for c in chunks if c.item_type == "list_item"]
        # 嵌套子项并入 parent；不拆
        assert len(list_chunks) == 1
        assert "outer" in list_chunks[0].content
        assert "inner" in list_chunks[0].content


class TestParagraphs:
    def test_paragraph_lines_merged_into_single_chunk(self):
        md = "## A\nThis is a paragraph.\nSecond line of same paragraph.\n\nNew paragraph.\n"
        chunks = chunk_markdown(md)
        para_chunks = [c for c in chunks if c.item_type == "paragraph"]
        assert len(para_chunks) == 2


class TestCodeBlocks:
    def test_code_block_becomes_single_chunk(self):
        md = "## A\n```python\nprint('hi')\n```\n"
        chunks = chunk_markdown(md)
        code_chunks = [c for c in chunks if c.item_type == "code_block"]
        assert len(code_chunks) == 1
        assert "print('hi')" in code_chunks[0].content


class TestContentHash:
    def test_hash_ignores_trailing_whitespace(self):
        a = chunk_markdown("- foo   \n")[0]
        b = chunk_markdown("- foo\n")[0]
        assert a.content_hash == b.content_hash
