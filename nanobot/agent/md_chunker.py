"""Markdown chunker: parse MEMORY.md into ordered structural chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+])\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```")


@dataclass
class Chunk:
    item_type: str           # 'heading' | 'list_item' | 'paragraph' | 'code_block'
    content: str
    content_hash: str
    section_path: str


def _normalize(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _build_path(stack: list[tuple[int, str]]) -> str:
    return " / ".join(t for _, t in stack)


def chunk_markdown(md: str) -> list[Chunk]:
    lines = md.splitlines()
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heading
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            path = _build_path(heading_stack)
            chunks.append(Chunk(
                item_type="heading", content=line.rstrip(),
                content_hash=_hash(line), section_path=path,
            ))
            i += 1
            continue

        # Code fence — accumulate until closing fence
        if _CODE_FENCE_RE.match(line):
            buf = [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if _CODE_FENCE_RE.match(lines[i]):
                    i += 1
                    break
                i += 1
            content = "\n".join(buf)
            chunks.append(Chunk(
                item_type="code_block", content=content,
                content_hash=_hash(content),
                section_path=_build_path(heading_stack),
            ))
            continue

        # List item (with potential nested children)
        m = _LIST_ITEM_RE.match(line)
        if m:
            base_indent = len(m.group(1))
            buf = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                m2 = _LIST_ITEM_RE.match(nxt)
                if m2 and len(m2.group(1)) <= base_indent:
                    break
                buf.append(nxt)
                i += 1
            content = "\n".join(buf).rstrip()
            chunks.append(Chunk(
                item_type="list_item", content=content,
                content_hash=_hash(content),
                section_path=_build_path(heading_stack),
            ))
            continue

        # Blank line — skip
        if not line.strip():
            i += 1
            continue

        # Paragraph — accumulate until blank / heading / list / code
        buf = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt) or _LIST_ITEM_RE.match(nxt) or _CODE_FENCE_RE.match(nxt):
                break
            buf.append(nxt)
            i += 1
        content = "\n".join(buf).rstrip()
        chunks.append(Chunk(
            item_type="paragraph", content=content,
            content_hash=_hash(content),
            section_path=_build_path(heading_stack),
        ))
    return chunks
