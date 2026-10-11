"""가중치 평균 (model soup) — 같은 구조 검사점 n 개의 가중치를 단순 평균해 새 검사점 폴더로 저장한다.

사용 (미니PC, 컨테이너 안) :
    run/podman-train.sh --name soup-a train/soup.py --out runs/soup-a runs/r2-02/ckpt-best runs/r2-06/ckpt-e4

- 입력 폴더마다 model.safetensors 를 읽는다. 키 집합 · 모양이 하나라도 다르면 멈춘다.
- 평균은 fp32 로 더한 뒤 n 으로 나누고, 첫 입력의 dtype 으로 되돌려 저장한다.
- 토크나이저 · config · model.json 따위 나머지 파일은 첫 입력 것을 복사한다.
  학습 상태(state.pt · state.json)와 best.json 은 평균 모델의 값이 아니라서 안 옮긴다.
- 평균한 원본 목록(경로 · model.safetensors SHA-256)은 soup.json 에 적는다.
같은 시작 가중치에서 갈라진 검사점끼리만 뜻이 있다 (다른 시작이면 평균이 망가진다) — 그건 부르는 쪽이 train_info.json 으로 확인한다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

WEIGHTS = "model.safetensors"
SKIP = {WEIGHTS, "state.pt", "state.json", "best.json", "soup.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def soup(inputs, out):
    from safetensors.torch import load_file, save_file

    if len(inputs) < 2:
        raise SystemExit("입력 검사점은 둘 이상이어야 한다")
    if os.path.exists(out):
        raise SystemExit(f"이미 있다 : {out}")
    for p in inputs:
        if not os.path.isfile(os.path.join(p, WEIGHTS)):
            raise SystemExit(f"{WEIGHTS} 가 없다 : {p}")
    # 하나씩 읽어 더한다 — 메모리는 입력 하나 + 누적 하나 몫
    import torch

    acc, first, keys = None, None, None
    sources = []
    for i, p in enumerate(inputs):
        wpath = os.path.join(p, WEIGHTS)
        sd = load_file(wpath)
        if first is None:
            first = {k: (v.dtype, tuple(v.shape)) for k, v in sd.items()}
            keys = set(sd)
            acc = {k: v.to(torch.float32).clone() for k, v in sd.items()}
        else:
            if set(sd) != keys:
                raise SystemExit(f"키 집합이 다르다 : {p} · {sorted(keys ^ set(sd))[:5]}")
            for k, v in sd.items():
                if tuple(v.shape) != first[k][1]:
                    raise SystemExit(f"모양이 다르다 : {p} · {k} {tuple(v.shape)} ≠ {first[k][1]}")
                acc[k] += v.to(torch.float32)
        sources.append({"path": p, "sha256": sha256(wpath)})
        del sd
        print(f"[soup] 더함 {i + 1}/{len(inputs)} : {p}", flush=True)
    n = len(inputs)
    merged = {k: (v / n).to(first[k][0]).contiguous() for k, v in acc.items()}

    tmp = out.rstrip("/") + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    save_file(merged, os.path.join(tmp, WEIGHTS), metadata={"format": "pt"})
    for name in sorted(os.listdir(inputs[0])):
        src = os.path.join(inputs[0], name)
        if name not in SKIP and os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp, name))
    info = {"method": "uniform mean (fp32 accumulate)", "n": n, "sources": sources,
            "copied_from": inputs[0], "weights_sha256": sha256(os.path.join(tmp, WEIGHTS))}
    with open(os.path.join(tmp, "soup.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    os.rename(tmp, out)
    print(f"[soup] 저장 : {out} · n={n}", flush=True)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="새 검사점 폴더 (없어야 한다)")
    p.add_argument("inputs", nargs="+", help="평균할 검사점 폴더들 (첫 것의 토크나이저 · config 를 복사)")
    args = p.parse_args(argv)
    soup(args.inputs, args.out)


if __name__ == "__main__":
    sys.exit(main())
