"""jira_dump — The Public Jira Dataset 의 mongodump 아카이브를 패키지 없이 흘려 읽어 얇은 jsonl 로 줄인다.

원자료는 `mongodump --gzip --archive` 한 덩이(압축 5.8GB · 풀면 약 60GB)라 통째로 풀지 않는다.
zip → gzip → 아카이브 → BSON 을 차례로 흘려 읽고, 이슈마다 쓸 칸만 남긴다(mine_public.py jira 가 읽는 꼴).

    python -I jira_dump.py probe --src <jira.zip> [--docs 2000]
    python -I jira_dump.py slim  --src <jira.zip> --out <issues.jsonl>

남기는 이슈 : 링크(issuelinks)가 하나라도 있거나, 해결 상태가 won't fix 류인 것.
남기는 칸 : key · project · summary · description(앞 3,000자) · components · resolution · issuelinks(종류 · 상대 key)
· comments(won't fix 류만, 앞 5개 · 각 1,000자). 받은 원자료는 자료로만 읽는다(안의 코드를 돌리지 않는다).
"""
import argparse
import collections
import gzip
import json
import struct
import sys
import zipfile

MAGIC = 0x8199E26D
TERMINATOR = 0xFFFFFFFF
DESC_CAP = 3000
COMMENT_CAP = 1000
COMMENT_COUNT = 5
WONTFIX = ("won't fix", "not a bug", "not a problem", "invalid", "won't do")

_FIXED = {0x01: 8, 0x06: 0, 0x07: 12, 0x08: 1, 0x09: 8, 0x0A: 0, 0x10: 4, 0x11: 8, 0x12: 8, 0x13: 16,
          0xFF: 0, 0x7F: 0}


# ---------------------------------------------------------------- BSON (필요한 만큼만)

def _cstring(buf, pos):
    end = buf.index(b"\x00", pos)
    return buf[pos:end].decode("utf-8", "replace"), end + 1


def _value(buf, kind, pos, want=None):
    """(값, 다음 자리). want 가 있으면 문서에서 그 이름만 푼다(나머지는 건너뜀)."""
    if kind in (0x02, 0x0D, 0x0E):
        size = struct.unpack_from("<i", buf, pos)[0]
        return buf[pos + 4:pos + 3 + size].decode("utf-8", "replace"), pos + 4 + size
    if kind in (0x03, 0x04):
        size = struct.unpack_from("<i", buf, pos)[0]
        doc = decode(buf, pos, want)
        if kind == 0x04:
            doc = list(doc.values())
        return doc, pos + size
    if kind == 0x05:
        size = struct.unpack_from("<i", buf, pos)[0]
        return None, pos + 5 + size
    if kind == 0x0F:
        size = struct.unpack_from("<i", buf, pos)[0]
        return None, pos + size
    if kind == 0x0B:
        _, pos = _cstring(buf, pos)
        _, pos = _cstring(buf, pos)
        return None, pos
    if kind == 0x0C:
        size = struct.unpack_from("<i", buf, pos)[0]
        return None, pos + 4 + size + 12
    if kind == 0x01:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if kind == 0x08:
        return buf[pos] == 1, pos + 1
    if kind == 0x10:
        return struct.unpack_from("<i", buf, pos)[0], pos + 4
    if kind == 0x12:
        return struct.unpack_from("<q", buf, pos)[0], pos + 8
    if kind in _FIXED:
        return None, pos + _FIXED[kind]
    raise ValueError(f"모르는 BSON 종류 0x{kind:02x}")


def _skip(buf, kind, pos):
    if kind in (0x02, 0x0D, 0x0E):
        return pos + 4 + struct.unpack_from("<i", buf, pos)[0]
    if kind in (0x03, 0x04, 0x0F):
        return pos + struct.unpack_from("<i", buf, pos)[0]
    if kind == 0x05:
        return pos + 5 + struct.unpack_from("<i", buf, pos)[0]
    if kind in _FIXED:
        return pos + _FIXED[kind]
    return _value(buf, kind, pos)[1]


def decode(buf, pos=0, want=None):
    """BSON 문서 하나. want = {이름: 아래 want 또는 None(통째)} — 없는 이름은 건너뛴다."""
    size = struct.unpack_from("<i", buf, pos)[0]
    end = pos + size - 1
    pos += 4
    out = {}
    while pos < end:
        kind = buf[pos]
        name, pos = _cstring(buf, pos + 1)
        if want is not None and name not in want:
            pos = _skip(buf, kind, pos)
            continue
        out[name], pos = _value(buf, kind, pos, want.get(name) if want is not None else None)
    return out


# ---------------------------------------------------------------- 아카이브 흘려 읽기

def open_stream(src):
    """zip 이면 안의 .archive 를, 아니면 파일 그대로 열어 gzip 을 벗긴 흐름."""
    if src.endswith(".zip"):
        archive = zipfile.ZipFile(src)
        member = [n for n in archive.namelist() if n.endswith(".archive")]
        if len(member) != 1:
            raise SystemExit(f"zip 안 .archive 가 하나가 아니다 : {member}")

        def opener():
            return archive.open(member[0])
    else:
        def opener():
            return open(src, "rb")
    with opener() as first:
        head = first.read(2)
    raw = opener()
    return gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("아카이브가 중간에 끊겼다")
    return data


def iter_docs(stream):
    """(컬렉션, BSON 바이트) 를 차례로. 앞머리(prelude)는 건너뛴다."""
    magic = struct.unpack("<I", _read_exact(stream, 4))[0]
    if magic != MAGIC:
        raise SystemExit(f"mongodump 아카이브 표지가 아니다 : 0x{magic:08x}")
    in_prelude, expect_header, collection = True, False, ""
    while True:
        head = stream.read(4)
        if not head:
            return
        if len(head) != 4:
            raise EOFError("아카이브가 중간에 끊겼다")
        size = struct.unpack("<I", head)[0]
        if size == TERMINATOR:
            in_prelude, expect_header = False, True
            continue
        if size < 5:  # BSON 최소 5바이트 — 깨진 길이로 read(음수) 가 흐름 끝까지 읽지 않게
            raise EOFError(f"BSON 길이가 깨졌다 : {size}")
        body = head + _read_exact(stream, size - 4)
        if in_prelude:
            continue
        if expect_header:
            header = decode(body, 0, {"db": None, "collection": None, "EOF": None})
            collection = header.get("collection", "")
            expect_header = False
            continue
        yield collection, body


# ---------------------------------------------------------------- 이슈 줄이기

_WANT = {
    "key": None,
    "fields": {
        "project": {"key": None},
        "summary": None,
        "description": None,
        "components": {str(i): {"name": None} for i in range(64)},
        "resolution": {"name": None},
        "issuelinks": {str(i): {"type": {"name": None}, "outwardIssue": {"key": None}, "inwardIssue": {"key": None}}
                       for i in range(512)},
        "comment": {"comments": {str(i): {"body": None} for i in range(COMMENT_COUNT)}},
    },
}


def slim_issue(doc, collection):
    fields = doc.get("fields") or {}
    links = []
    for link in fields.get("issuelinks") or []:
        if not isinstance(link, dict):
            continue
        kind = (link.get("type") or {}).get("name", "")
        for side in ("outwardIssue", "inwardIssue"):
            other = (link.get(side) or {}).get("key")
            if other:
                links.append({"type": {"name": kind}, side: {"key": other}})
    resolution = ((fields.get("resolution") or {}).get("name") or "")
    if not links and resolution.lower() not in WONTFIX:
        return None
    comments = []
    if resolution.lower() in WONTFIX:
        comments = [{"body": (c.get("body") or "")[:COMMENT_CAP]}
                    for c in ((fields.get("comment") or {}).get("comments") or []) if isinstance(c, dict)]
    return {
        "key": doc.get("key", ""),
        "repo": collection,
        "fields": {
            "project": {"key": (fields.get("project") or {}).get("key", "")},
            "summary": fields.get("summary") or "",
            "description": (fields.get("description") or "")[:DESC_CAP],
            "components": [{"name": c.get("name", "")} for c in fields.get("components") or [] if isinstance(c, dict)],
            "resolution": {"name": resolution} if resolution else None,
            "issuelinks": links,
            "comment": {"comments": comments},
        },
    }


def cmd_probe(args):
    counts = collections.Counter()
    shown = set()
    for index, (collection, body) in enumerate(iter_docs(open_stream(args.src))):
        if index >= args.docs:
            break
        counts[collection] += 1
        if collection not in shown:
            shown.add(collection)
            keys = []
            pos = 4
            while pos < len(body) - 1:
                kind = body[pos]
                name, pos = _cstring(body, pos + 1)
                keys.append(name)
                pos = _skip(body, kind, pos)
            print(json.dumps({"collection": collection, "bytes": len(body), "top_keys": keys[:30]}, ensure_ascii=False))
    print(json.dumps({"first_docs": dict(counts)}, ensure_ascii=False))
    return 0


def cmd_slim(args):
    seen = collections.Counter()
    kept = collections.Counter()
    link_kinds = collections.Counter()
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for collection, body in iter_docs(open_stream(args.src)):
            seen[collection] += 1
            try:
                doc = decode(body, 0, _WANT)
            except (ValueError, struct.error, IndexError):
                seen["!bad"] += 1
                continue
            issue = slim_issue(doc, collection)
            if issue is None or not issue["key"]:
                continue
            kept[collection] += 1
            for link in issue["fields"]["issuelinks"]:
                link_kinds[link["type"]["name"]] += 1
            out.write(json.dumps(issue, ensure_ascii=False) + "\n")
            if args.limit and sum(kept.values()) >= args.limit:
                break
    print(json.dumps({"seen": dict(seen), "kept": dict(kept), "kept_total": sum(kept.values()),
                      "link_kinds_top": dict(link_kinds.most_common(25))}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Public Jira mongodump 아카이브 → 얇은 이슈 jsonl")
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="앞 문서 몇 개로 컬렉션 · 칸 이름만 본다")
    probe.add_argument("--src", required=True, help="jira.zip 또는 .archive")
    probe.add_argument("--docs", type=int, default=2000)
    slim = sub.add_parser("slim", help="링크 · won't fix 이슈만 얇게 쓴다")
    slim.add_argument("--src", required=True, help="jira.zip 또는 .archive")
    slim.add_argument("--out", required=True)
    slim.add_argument("--limit", type=int, default=0, help="남긴 이슈 수 상한(시험용, 0 = 끝까지)")
    args = parser.parse_args(argv)
    return cmd_probe(args) if args.cmd == "probe" else cmd_slim(args)


if __name__ == "__main__":
    sys.exit(main())
