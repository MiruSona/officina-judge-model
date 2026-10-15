"""restamp — 기준선 평가 폴더를 새 규약 해시로 「다시 찍은 보기」를 만든다 (원천 균형 판 설계 5-3).

규약 판이 바뀌면(g5 → g6) 옛 판 summary.json 의 protocol_sha256 이 달라 compare.py 가 exit 2 로 거부한다.
eval_nli.py 는 --protocol 을 summary 에 해시를 찍는 데만 쓰므로(예측에 안 닿는다) GPU 로 다시 잴 까닭이 없다.
그래서 판 폴더를 그대로 본뜬 새 폴더를 만들고, 파일은 원본 링크로 · summary.json 만 새 해시로 바꿔 쓴다.

    python3 -I restamp.py --protocol src/measure/protocol-g6.json --root runs --out runs/g4-K4-eval/restamp \
        --runs g1-K-s1/g4 g1-K-s2/g4 … g1-K3-s5/g4 [--note "까닭"]
    → <out>/g1-K-s1/g4/… (pred · calibration · eval-side 는 원본 링크, summary.json 만 다시 찍음) · <out>/restamp.json

다시 찍은 summary 에는 옛 값(restamp_from_protocol_sha256)과 까닭(restamp_note) 칸을 덧붙인다 — K3 때 손으로 한 꼴과 같다.
원본 판 폴더 안에 쓰거나(--out 이 판 폴더 안) 이미 있는 판 폴더를 덮는 것은 거부한다(exit 2).
링크는 심볼릭 링크, 못 만들면(윈도 권한) 하드 링크. 링크한 파일은 열어 쓰지 않는다.
판 폴더 안 디렉터리 심볼릭 링크는 따라 들어가 그 안 파일을 하나씩 링크한다.
한 판이 중간에 실패해도 그때까지 한 판(실패한 판의 앞부분 포함)과 실패 까닭(failed)을 restamp.json 에 남기고 exit 2 —
실패한 판의 보기 폴더는 지우고 다시 돌린다(이미 있으면 덮지 않는다).
"""
import argparse
import hashlib
import json
import os
import sys

REFUSE_EXIT = 2
NOTE = "규약 판이 바뀌어 summary 의 protocol_sha256 만 새 규약으로 다시 찍었다 — 예측 · 문턱 · 곁 자는 원본 그대로 (restamp.py)"


class Refuse(Exception):
    """쓰면 안 되는 자리 — main 이 exit 2 로 바꾼다."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _inside(path, folder):
    """path 가 folder 자신이거나 그 안인가 (실제 경로로 견준다)."""
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return os.path.normcase(path) == os.path.normcase(folder) or \
        os.path.normcase(path).startswith(os.path.normcase(folder.rstrip(os.sep) + os.sep))


def link(src, dst):
    """dst 에 src 링크 → 「symlink」 · 「hardlink」. 둘 다 못 하면 Refuse (복사는 안 한다 — 원본과 갈라진다)."""
    try:
        os.symlink(os.path.abspath(src), dst)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e:
        raise Refuse(f"링크를 못 만든다 {dst} : {e}") from e


def restamp_summary(summary, new_sha, note=NOTE):
    """summary dict → 다시 찍은 dict. 옛 값 · 까닭 칸을 덧붙이고 그 밖 칸 · 차례는 그대로."""
    out = dict(summary)
    old = summary.get("protocol_sha256")
    out["protocol_sha256"] = new_sha
    out["restamp_from_protocol_sha256"] = old
    out["restamp_note"] = note
    return out


def plan_checks(root, out, runs):
    """쓰기 전 검사 — 판 폴더가 있고, --out 이 어느 판 폴더 · 그 윗 판 폴더 안도 아니고, 나갈 자리가 비었나."""
    for rel in runs:
        src = os.path.join(root, rel)
        if not os.path.isdir(src):
            raise Refuse(f"판 폴더가 없다 {src}")
        top = os.path.join(root, rel.replace("\\", "/").split("/")[0])
        if _inside(out, top) or _inside(out, src):
            raise Refuse(f"--out 이 원본 판 폴더 안이다 {out} — 원본에는 안 쓴다")
        if _inside(src, out):
            raise Refuse(f"판 폴더 {src} 가 --out 안에 있다 — 원본과 보기 폴더를 가른다")
        dst = os.path.join(out, rel)
        if os.path.exists(dst):
            raise Refuse(f"이미 있다 {dst} — 덮지 않는다")


def restamp_run(src, dst, new_sha, note, done=None):
    """판 폴더 하나를 본뜬다 → [{file, kind, old_protocol_sha256?}]. summary.json 만 새로 쓰고 나머지는 링크.
    done 을 주면 그 목록에 이어 적는다 — 중간에 실패해도 부른 쪽이 한 데까지를 안다."""
    done = [] if done is None else done
    for here, dirs, files in os.walk(src, followlinks=True):
        dirs.sort()
        rel = os.path.relpath(here, src)
        target = os.path.normpath(os.path.join(dst, rel))
        os.makedirs(target, exist_ok=True)
        for name in sorted(files):
            path, out_path = os.path.join(here, name), os.path.join(target, name)
            entry = {"file": os.path.normpath(os.path.join(rel, name)).replace("\\", "/")}
            if name == "summary.json":
                with open(path, encoding="utf-8") as f:
                    summary = json.load(f)
                with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(restamp_summary(summary, new_sha, note), f, ensure_ascii=False, indent=1)
                    f.write("\n")
                entry.update(kind="restamped", old_protocol_sha256=summary.get("protocol_sha256"))
            else:
                entry["kind"] = link(path, out_path)
            done.append(entry)
    return done


def restamp(protocol, root, out, runs, note=NOTE):
    """판 여럿을 다시 찍는다 → 보고 dict (<out>/restamp.json 에도 쓴다)."""
    plan_checks(root, out, runs)
    new_sha = sha256_file(protocol)
    path = os.path.join(out, "restamp.json")
    before = {}
    if os.path.exists(path):                 # 같은 보기 폴더에 판을 나눠 더할 때 — 앞 기록을 잇는다
        with open(path, encoding="utf-8") as f:
            before = json.load(f)
        if before.get("protocol_sha256") != new_sha:
            raise Refuse(f"{path} 의 규약 해시가 다르다 — 보기 폴더 하나에 규약 하나")
    report = {"protocol": os.path.basename(protocol), "protocol_sha256": new_sha, "note": note,
              "runs": dict(before.get("runs", {}))}
    failed = None
    for rel in runs:
        report["runs"][rel] = []
        try:
            restamp_run(os.path.join(root, rel), os.path.join(out, rel), new_sha, note, report["runs"][rel])
        except (Refuse, OSError, ValueError) as e:     # 한 판이 실패해도 그때까지 한 것은 restamp.json 에 남긴다
            failed = (rel, e)
            report["failed"] = {rel: str(e)}
            break
    report["summaries"] = sum(1 for files in report["runs"].values() for e in files if e["kind"] == "restamped")
    os.makedirs(out, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if failed:
        rel, e = failed
        raise Refuse(f"{rel} 판에서 멈췄다 : {e} — 한 데까지는 {path} 에 적었다. 그 판 보기 폴더를 지우고 다시 돌린다") from e
    return report


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="기준선 summary 를 새 규약 해시로 다시 찍은 보기 폴더")
    p.add_argument("--protocol", required=True, help="새 규약 (protocol-g6.json) — 이 파일 sha256 을 찍는다")
    p.add_argument("--root", required=True, help="판 폴더들의 뿌리 (runs)")
    p.add_argument("--out", required=True, help="보기 폴더 뿌리 (runs/g4-K4-eval/restamp)")
    p.add_argument("--runs", nargs="+", required=True, help="--root 기준 판 폴더 (g1-K-s1/g4 …)")
    p.add_argument("--note", default=NOTE, help="summary 에 덧붙일 까닭")
    a = p.parse_args(argv)
    try:
        report = restamp(a.protocol, a.root, a.out, a.runs, a.note)
    except Refuse as e:
        print(f"거부: {e}", file=sys.stderr)
        return REFUSE_EXIT
    print(json.dumps({"protocol_sha256": report["protocol_sha256"], "runs": len(report["runs"]),
                      "summaries": report["summaries"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
