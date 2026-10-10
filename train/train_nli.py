"""한국어 NLI 3갈래 분류기 학습 고리 — 기반 학습과 덧학습 공용 (설계 3절 · 4절).

    # 기반 학습 (nli-base-v1)
    python -I train_nli.py --device cuda --dtype bf16 --init /work/hf/xlm-roberta-large \
        --train data/prep/klue-train.jsonl --train-epoch data/prep/kornli-e{epoch}.jsonl \
        --train-epoch data/prep/mnli-e{epoch}.jsonl --train-epoch data/prep/snli-e{epoch}.jsonl \
        --dev data/prep/klue-dev.jsonl --out runs/base-2 --epochs 2 --batch 32 --max-len 128
    # 덧학습 (nli-studio-v1) — mem 쌍 + 같은 수 KLUE replay · 갈래 가중치 · 바퀴 끝마다 dev
    python -I train_nli.py --device cuda --dtype bf16 --init runs/base-1/ckpt-best \
        --train mem/train.jsonl --replay data/prep/klue-train.jsonl --mix-klue 1:1 --class-weight \
        --dev mem/dev.jsonl --out runs/studio-1 --epochs 4 --batch 16 --warmup 0.10 --eval-every 0 \
        --select last --collapse-acc 0 --max-len 256
    # 스크립트가 끝까지 도는지만 (내려받기 없음 · 무작위 초기화 작은 모델 · CPU)
    python -I train_nli.py --smoke --out <임시 폴더>

datasets · accelerate 없이 torch 와 transformers(AutoTokenizer · AutoModelForSequenceClassification)만 쓴다.
멈춤 : NaN·inf 3 연속 → nan · 스왑 +256MB → swap · 멈춤 파일(JT_STOP_FILE) → 그 안의 까닭.
게임 서버 · 「게임 서버가 있을 때 가용 < 10GB」는 컨테이너 안에서 못 본다(pid 이름공간이 다르다) — 감싸개
(run/podman-train.sh · run/cpu-train.sh)가 보고 멈춤 파일을 쓴다. 심장 박동 60초 끊김도 감싸개 몫이다.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from data_prep import LABELS, LETTERS, LICENSES, SOURCE_LABELS, read_pairs, sha256_file  # noqa: E402
from pairenc import check_no_token_types, encode_pair  # noqa: E402

EXIT_GUARD = 3        # 멈춤 파일 · 스왑 — 검사점 저장 뒤 멈춤
EXIT_NAN = 4
EXIT_COLLAPSE = 5
NAN_LIMIT = 3
GROUP_CHUNK = 50      # 길이별 묶음 : 배치 50 개 몫을 길이로 정렬해 자른다
LOG_EVERY = 20
MIB = 1024 * 1024
SMOKE_ROWS = 64
SMOKE_MAX_LEN = 64

_log_file = None


class Collapse(Exception):
    """large 가 한 라벨만 내는 쪽으로 무너졌다 (설계 3-2)."""


class Stop(Exception):
    """밖에서 멈추라고 했거나(멈춤 파일 · 스왑) NaN 이 이어졌다."""

    def __init__(self, reason, code):
        super().__init__(reason)
        self.reason = reason
        self.code = code


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JudgeModel NLI 학습 (기반 · 덧학습)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="bf16 은 autocast (가중치·옵티마이저 fp32)")
    p.add_argument("--init", default="", help="시작 모델 폴더 또는 HF id (xlm-roberta-large · 앞 판 ckpt-best)")
    p.add_argument("--new-head", action="store_true", help="머리 칸 수가 다른 모델에서 머리만 새로 (지름길)")
    p.add_argument("--stage", choices=["base", "studio"], default="base")
    p.add_argument("--train", action="append", default=[], help="바퀴마다 다 쓰는 쌍 jsonl (여러 번)")
    p.add_argument("--train-epoch", action="append", default=[],
                   help="바퀴마다 다른 파일 — {epoch} 자리에 1부터 바퀴 번호 (KorNLI 표본)")
    p.add_argument("--replay", default="", help="잊음 막기 replay 풀 (KLUE 학습 jsonl)")
    p.add_argument("--mix-klue", default="1:1", help="--train 쌍 : replay 쌍 비율 (2:1 = replay 를 반으로)")
    p.add_argument("--class-weight", action="store_true", help="mem 쌍에만 갈래 수의 역수 가중 (있는 갈래들의 가중치 단순 평균 1)")
    p.add_argument("--dev", default="", help="평가 셋 jsonl")
    p.add_argument("--dev-limit", type=int, default=3000)
    p.add_argument("--out", required=True, help="판 폴더 runs/<run_id>")
    p.add_argument("--resume", action="store_true", help="ckpt-last 의 바퀴·스텝 위치부터 잇는다")
    p.add_argument("--smoke", action="store_true", help="64쌍 · 무작위 초기화 작은 모델 · CPU · 저장·재개 확인")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=1, help="기울기 누적 (CPU 는 16 × 2)")
    p.add_argument("--warmup", type=float, default=0.06, help="전체 스텝 중 warmup 몫")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--betas", default="0.9,0.98")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=2000, help="스텝마다 dev (0 = 바퀴 끝만)")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--ckpt-minutes", type=float, default=20)
    p.add_argument("--select", choices=["base", "studio", "last"], default=None,
                   help="ckpt-best 고르기. base = dev 정확도 최고 · studio = 거짓 지지 최소 → 갈래 재현 최저값 최고 "
                        "(덜 배운 바퀴가 유리한 함정 — better() 참고) · last = 마지막 바퀴. "
                        "기본은 --stage base 면 base, studio 면 last")
    p.add_argument("--collapse-acc", type=float, default=0.50, help="첫 검사 dev 정확도가 이 아래면 무너짐 (0 = 끔)")
    p.add_argument("--collapse-step", type=int, default=2000)
    p.add_argument("--collapse-retries", type=int, default=2, help="씨앗 +1 로 다시 (그 뒤 lr 5e-6 한 번)")
    p.add_argument("--collapse-lr", type=float, default=5e-6)
    p.add_argument("--max-steps", type=int, default=0, help="이 스텝에서 검사점 저장 뒤 멈춤 (시험용)")
    p.add_argument("--swap-max-mb", type=int, default=256, help="스왑이 시작보다 이만큼 늘면 멈춤 (0 = 끔)")
    p.add_argument("--heartbeat", default=os.environ.get("JT_HEARTBEAT", ""))
    p.add_argument("--stop-file", default=os.environ.get("JT_STOP_FILE", ""),
                   help="이 파일이 생기면 검사점 저장 뒤 멈춘다 (감싸개가 게임 서버·메모리를 보고 쓴다)")
    p.add_argument("--run-id", default="")
    args = p.parse_args(argv)
    if args.select is None:
        args.select = "last" if args.stage == "studio" else "base"
    return args


# ---------------------------------------------------------------- 자료

def load_rows(path):
    """라벨 없는 줄은 버린다. (줄, 버린 수)."""
    rows = read_pairs(path)
    kept = [r for r in rows if r.get("label") in (0, 1, 2)]
    return kept, len(rows) - len(kept)


def parse_ratio(text):
    main_part, replay_part = text.split(":", 1)
    return float(main_part), float(replay_part)


def replay_count(args, n_main):
    if not args.replay:
        return 0
    main_part, replay_part = parse_ratio(args.mix_klue)
    return int(round(n_main * replay_part / main_part))


def class_weights(rows):
    """mem 쌍 갈래 수의 역수로 만든 [wa, wb, wc].

    「갈래 평균 1」 = mem 쌍이 하나라도 있는 갈래들의 가중치를 단순 평균하면 1 이다 (쌍 수로 가중한 평균이 아니다).
    mem 쌍이 없는 갈래는 0 이고 평균에서 뺀다. mem 쌍이 아예 없으면 모두 1.
    """
    counts = [0, 0, 0]
    for r in rows:
        if r["src"] == "mem":
            counts[r["label"]] += 1
    present = [c for c in counts if c]
    if not present:
        return [1.0, 1.0, 1.0]
    inv = [1.0 / c if c else 0.0 for c in counts]
    mean = sum(inv) / len(present)
    return [w / mean for w in inv]


def tokenize(rows, backend, max_len, weights):
    """줄 → 학습 항목 [(ids, label, 가중, kind)] · 잘린 쌍 수."""
    items, truncated = [], 0
    for r in rows:
        ids, cut = encode_pair(backend, r["premise"], r["hypothesis"], max_len)
        truncated += 1 if cut else 0
        w = 1.0
        if weights is not None and r["src"] == "mem":
            w = weights[r["label"]]
        items.append((ids, r["label"], w, r.get("kind", "")))
    return items, truncated


def make_batches(items, batch, seed, epoch):
    """길이별 묶음. 씨앗·바퀴로 차례가 정해져 재개 때 같은 차례가 나온다."""
    rng = random.Random(seed * 1000 + epoch)
    order = list(range(len(items)))
    rng.shuffle(order)
    chunk = batch * GROUP_CHUNK
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: len(items[i][0]))
        for b in range(0, len(part), batch):
            batches.append(part[b:b + batch])
    rng.shuffle(batches)
    return batches


def collate(items, idxs, pad_id, device):
    width = max(len(items[i][0]) for i in idxs)
    ids = torch.full((len(idxs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idxs), width), dtype=torch.long)
    for row, i in enumerate(idxs):
        seq = items[i][0]
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, :len(seq)] = 1
    labels = torch.tensor([items[i][1] for i in idxs], dtype=torch.long)
    weights = torch.tensor([items[i][2] for i in idxs], dtype=torch.float)
    return ids.to(device), mask.to(device), labels.to(device), weights.to(device)


def data_entry(path, rows, truncated):
    by_label = {LABELS[i]: sum(1 for r in rows if r["label"] == i) for i in range(3)}
    by_src = {}
    for r in rows:
        by_src[r["src"]] = by_src.get(r["src"], 0) + 1
    licenses = sorted({LICENSES.get(s, ("unknown", True))[0] for s in by_src})
    nc = any(LICENSES.get(s, ("unknown", True))[1] for s in by_src)
    return {"file": os.path.basename(path), "sha256": sha256_file(path), "rows": len(rows),
            "by_label": by_label, "by_src": by_src, "license": licenses, "nc": nc, "truncated": truncated}


# ---------------------------------------------------------------- 모델

def build_model(init, device, new_head):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        init, num_labels=3, ignore_mismatched_sizes=new_head,
        id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)})
    check_no_token_types(model.config)
    return model.to(device)


def load_tokenizer(init):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(init, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("fast 토크나이저가 아니다 — 서버(tokenizers)와 같은 셈을 보장할 수 없다")
    return tokenizer


def make_optimizer(model, args, lr):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    betas = tuple(float(x) for x in args.betas.split(","))
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=args.eps)


def make_scheduler(optimizer, total_steps, warmup_share):
    warm = max(1, int(total_steps * warmup_share))

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def forward_loss(model, batch, device, use_bf16):
    ids, mask, labels, weights = batch
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids=ids, attention_mask=mask).logits
    losses = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
    return (losses * weights).mean()


def evaluate(model, items, batch, pad_id, device, use_bf16, want_logits=False):
    """정확도 · 갈래별 재현 · 거짓 지지(정답 B·C 를 A 로) · 혼동표[정답][예측] · 평균 손실 · 지지 kind 별 재현."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    logits_by = [None] * len(items)
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(order), batch):
            idxs = order[start:start + batch]
            ids, mask, labels, _ = collate(items, idxs, pad_id, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=ids, attention_mask=mask).logits.float()
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            for k, i in enumerate(idxs):
                logits_by[i] = logits[k].tolist()
    model.train()
    metrics = metrics_from(logits_by, [it[1] for it in items], [it[3] for it in items])
    metrics["loss"] = round(loss_sum / max(1, len(items)), 5)
    if want_logits:
        return metrics, logits_by
    return metrics


def metrics_from(logits, labels, kinds):
    confusion = [[0, 0, 0] for _ in range(3)]
    for lg, gold in zip(logits, labels):
        pred = max(range(3), key=lambda i: lg[i])
        confusion[gold][pred] += 1
    n = max(1, len(labels))
    recall = {}
    for i in range(3):
        total = sum(confusion[i])
        recall[LETTERS[i]] = round(confusion[i][i] / total, 4) if total else None
    by_kind = {}
    for lg, gold, kind in zip(logits, labels, kinds):
        if gold != 0 or not kind:
            continue
        hit, total = by_kind.get(kind, (0, 0))
        by_kind[kind] = (hit + (1 if max(range(3), key=lambda i: lg[i]) == 0 else 0), total + 1)
    return {"n": len(labels), "acc": round(sum(confusion[i][i] for i in range(3)) / n, 4), "recall": recall,
            "false_support": confusion[1][0] + confusion[2][0], "confusion": confusion,
            "support_recall_by_kind": {k: round(h / t, 4) for k, (h, t) in sorted(by_kind.items())}}


def better(select, new, old):
    """새 평가가 지금 최고보다 나은가 (설계 3-2 · 4-2).

    last 는 늘 참이다 — 마지막 평가가 best 가 된다. 저장은 run_eval 이 건너뛰고 학습 끝에 한 번만 한다.
    studio 의 함정 : 온도 전 거짓 지지 「개수」를 먼저 견주므로, 지지를 덜 내는 덜 배운 바퀴(1~2 바퀴째)가
    구조적으로 유리하다. 길 1 보강 반복 다섯 바퀴 중 네 번 이른 바퀴를 골랐다
    (Docs/Research/2026-10-09-길1보강반복.md). 그래서 덧학습 기본은 last 다.
    """
    if old is None or select == "last":
        return True
    if select == "base":
        return new["acc"] > old["acc"]
    new_min = min(v for v in new["recall"].values() if v is not None)
    old_min = min(v for v in old["recall"].values() if v is not None)
    return (-new["false_support"], new_min) > (-old["false_support"], old_min)


# ---------------------------------------------------------------- 검사점 · 기록

def save_dir_atomic(path, write):
    tmp = path + ".tmp"
    old = path + ".old"
    for leftover in (tmp, old):
        if os.path.isdir(leftover):
            shutil.rmtree(leftover)
    os.makedirs(tmp)
    write(tmp)
    if os.path.isdir(path):
        os.replace(path, old)
    os.replace(tmp, path)
    if os.path.isdir(old):
        shutil.rmtree(old)


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state):
    # encode_pair 가 건 자르기 설정이 tokenizer.json 에 박히지 않게 푼다
    tokenizer.backend_tokenizer.no_truncation()

    def write(tmp):
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        blob = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "state": state}
        if torch.cuda.is_available():
            blob["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(blob, os.path.join(tmp, "state.pt"))
        with open(os.path.join(tmp, "state.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)

    save_dir_atomic(path, write)


def save_best(path, model, tokenizer, metrics, state):
    tokenizer.backend_tokenizer.no_truncation()

    def write(tmp):
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        with open(os.path.join(tmp, "best.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"epoch": state["epoch"], "step": state["step"], "dev": metrics}, f, ensure_ascii=False, indent=1)

    save_dir_atomic(path, write)


def load_checkpoint(path):
    return torch.load(os.path.join(path, "state.pt"), map_location="cpu", weights_only=False)


def write_heartbeat(path, step, loss):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "step": step, "loss": loss}, f)
    os.replace(tmp, path)


def read_meminfo():
    """(MemAvailable, 스왑 사용) MiB. 컨테이너 안에서도 호스트 값이다. 없으면 (None, None)."""
    if not os.path.exists("/proc/meminfo"):
        return None, None
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            vals[key] = int(rest.split()[0])
    return vals["MemAvailable"] // 1024, (vals["SwapTotal"] - vals["SwapFree"]) // 1024


def read_gtt_used():
    import glob

    total, seen = 0, False
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        with open(path) as f:
            total += int(f.read().strip())
        seen = True
    if not seen:
        return None
    return total // MIB


def peak_rss_mb():
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


class Watch:
    """메모리 기록 · 스왑 멈춤 · 멈춤 파일. 매 옵티마이저 스텝마다 check() 를 부른다."""

    def __init__(self, args):
        self.args = args
        avail, swap = read_meminfo()
        self.swap_start = swap
        self.min_avail = avail
        self.swap_max = swap
        self.gtt_peak = read_gtt_used()

    def check(self):
        avail, swap = read_meminfo()
        if avail is not None:
            self.min_avail = min(self.min_avail, avail)
            self.swap_max = max(self.swap_max, swap)
        gtt = read_gtt_used()
        if gtt is not None:
            self.gtt_peak = max(self.gtt_peak or 0, gtt)
        stop_file = self.args.stop_file
        if stop_file and os.path.exists(stop_file):
            with open(stop_file, encoding="utf-8") as f:
                reason = f.read().strip() or "external"
            raise Stop(reason, EXIT_GUARD)
        if self.args.swap_max_mb and swap is not None and swap - self.swap_start > self.args.swap_max_mb:
            raise Stop(f"swap (+{swap - self.swap_start}MB)", EXIT_GUARD)

    def summary(self):
        delta = None
        if self.swap_start is not None:
            delta = self.swap_max - self.swap_start
        return {"peak_rss_mb": peak_rss_mb(), "peak_gtt_mb": self.gtt_peak,
                "min_mem_available_mb": self.min_avail, "swap_delta_mb": delta}


def versions():
    import tokenizers
    import transformers

    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}


def script_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_info(args, info):
    path = os.path.join(args.out, "train_info.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- 학습 한 판

class Run:
    """한 판의 자료 · 모델 · 상태. 무너짐으로 다시 시작하면 새로 만든다."""

    def __init__(self, args, seed, lr, restarts):
        self.args = args
        self.device = args.device
        self.use_bf16 = args.dtype == "bf16"
        last = os.path.join(args.out, "ckpt-last")
        self.resume_blob = None
        init = args.init
        if args.resume:
            if not os.path.isdir(last):
                raise SystemExit(f"--resume 인데 검사점이 없다: {last}")
            self.resume_blob = load_checkpoint(last)
            init = last
        self.tokenizer = load_tokenizer(init)
        self.backend = self.tokenizer.backend_tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.model = build_model(init, self.device, args.new_head and not args.resume)
        self.model.train()
        self.load_data()
        state = {"epoch": 0, "micro": 0, "step": 0, "seed": seed, "lr": lr, "restarts": restarts,
                 "best": None, "best_metrics": None, "dev_history": [], "started": time.time()}
        if self.resume_blob is not None:
            state = self.resume_blob["state"]
        self.state = state
        self.total_steps = self.count_steps()
        self.optimizer = make_optimizer(self.model, args, state["lr"])
        self.scheduler = make_scheduler(self.optimizer, self.total_steps, args.warmup)
        if self.resume_blob is not None:
            self.optimizer.load_state_dict(self.resume_blob["optimizer"])
            self.scheduler.load_state_dict(self.resume_blob["scheduler"])
            torch.set_rng_state(self.resume_blob["torch_rng"])
            if "cuda_rng" in self.resume_blob and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.resume_blob["cuda_rng"])
        else:
            torch.manual_seed(state["seed"])
        self.watch = Watch(args)

    def load_data(self):
        args = self.args
        self.main_rows, self.data = [], []
        self.dropped = 0
        for path in args.train:
            rows, dropped = load_rows(path)
            self.dropped += dropped
            self.main_rows.extend(rows)
            self.data.append((path, rows))
        self.weights = None
        if args.class_weight:
            self.weights = class_weights(self.main_rows)
        self.main_items, self.truncated = [], {}
        for path, rows in self.data:
            items, cut = tokenize(rows, self.backend, args.max_len, self.weights)
            self.main_items.extend(items)
            self.truncated[os.path.basename(path)] = cut
        self.replay_pool = []
        if args.replay:
            self.replay_pool, dropped = load_rows(args.replay)
            self.dropped += dropped
        self.dev_items = []
        if args.dev:
            dev_rows, _ = load_rows(args.dev)
            dev_rows = dev_rows[:args.dev_limit]
            self.dev_items, self.truncated["_dev"] = tokenize(dev_rows, self.backend, args.max_len, None)

    def epoch_extra_paths(self, epoch):
        return [t.replace("{epoch}", str(epoch + 1)) for t in self.args.train_epoch]

    def epoch_items(self, epoch):
        """바퀴 하나의 항목 = --train 전부 + 이 바퀴 파일 + replay 표본."""
        items = list(self.main_items)
        for path in self.epoch_extra_paths(epoch):
            rows, _ = load_rows(path)
            extra, cut = tokenize(rows, self.backend, self.args.max_len, None)
            self.truncated[os.path.basename(path)] = cut
            items.extend(extra)
        n_replay = replay_count(self.args, len(self.main_rows))
        if n_replay:
            rng = random.Random(self.state["seed"] * 7919 + epoch)
            picked = rng.sample(self.replay_pool, min(n_replay, len(self.replay_pool)))
            extra, _ = tokenize(picked, self.backend, self.args.max_len, None)
            items.extend(extra)
        return items

    def count_steps(self):
        """epoch_items 와 같은 셈 — 라벨 있는 줄만 · replay 는 풀 크기까지. 어긋나면 lr 이 0 까지 안 내려간다."""
        args = self.args
        total = 0
        n_replay = min(replay_count(args, len(self.main_rows)), len(self.replay_pool))
        for epoch in range(args.epochs):
            n = len(self.main_rows) + n_replay
            for path in self.epoch_extra_paths(epoch):
                n += len(load_rows(path)[0])
            micro = math.ceil(n / args.batch)
            total += math.ceil(micro / args.accum)
        return max(1, total)

    def info(self, stop_reason):
        args = self.args
        data = [data_entry(p, rows, self.truncated.get(os.path.basename(p), 0)) for p, rows in self.data]
        for epoch in range(args.epochs):
            for path in self.epoch_extra_paths(epoch):
                if os.path.exists(path):
                    rows, _ = load_rows(path)
                    data.append(data_entry(path, rows, self.truncated.get(os.path.basename(path), 0)))
        if args.replay:
            entry = data_entry(args.replay, self.replay_pool, 0)
            per_epoch = min(replay_count(args, len(self.main_rows)), len(self.replay_pool))
            entry["replay"] = {"mix": args.mix_klue, "per_epoch": per_epoch}
            data.append(entry)
        st = self.state
        elapsed = max(1e-9, time.time() - st["started"])
        info = {
            "run_id": args.run_id or os.path.basename(os.path.abspath(args.out)),
            "stage": args.stage, "init": args.init, "device": args.device, "dtype": args.dtype,
            "aotriton": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
            "versions": versions(), "image": os.environ.get("JT_IMAGE_DIGEST") or os.environ.get("JT_IMAGE") or "venv",
            "data": data, "dropped_no_label": self.dropped, "max_len": args.max_len,
            "dev": {"file": os.path.basename(args.dev), "rows": len(self.dev_items),
                    "truncated": self.truncated.get("_dev", 0)} if args.dev else None,
            "labels": LABELS, "source_labels": SOURCE_LABELS,
            "hparams": {"lr": st["lr"], "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
                        "warmup": args.warmup, "wd": args.wd, "eps": args.eps, "betas": args.betas,
                        "eval_every": args.eval_every, "select": args.select, "class_weight": self.weights,
                        "mix_klue": args.mix_klue if args.replay else None, "new_head": args.new_head,
                        "total_steps": self.total_steps},
            "seed": st["seed"], "restarts": st["restarts"], "steps": st["step"],
            "sec_per_step": round(elapsed / max(1, st["step"]), 4),
            "pairs_per_min": round(st["step"] * args.batch * args.accum / elapsed * 60, 1),
            **self.watch.summary(),
            "dev_history": st["dev_history"], "best": st["best"],
            "stop_reason": stop_reason, "script_sha": script_sha(),
        }
        return info

    def checkpoint(self):
        save_checkpoint(os.path.join(self.args.out, "ckpt-last"), self.model, self.tokenizer,
                        self.optimizer, self.scheduler, self.state)
        self.last_ckpt = time.time()

    def run_eval(self, epoch_end):
        if not self.dev_items:
            return None
        history = self.state["dev_history"]
        if epoch_end and history and history[-1]["step"] == self.state["step"]:
            # 방금 같은 스텝에서 쟀다 — 바퀴 끝 표시만 고친다
            history[-1]["epoch"] = self.state["epoch"] + 1
            best = self.state["best"]
            if best and best["step"] == self.state["step"]:
                best["epoch"] = self.state["epoch"] + 1
            return None
        metrics = evaluate(self.model, self.dev_items, self.args.batch * 2, self.pad_id, self.device, self.use_bf16)
        st = self.state
        row = {"epoch": st["epoch"] + (1 if epoch_end else 0), "step": st["step"], **metrics}
        st["dev_history"].append(row)
        log(f"dev epoch {row['epoch']} step {st['step']} acc {metrics['acc']} recall {metrics['recall']} "
            f"false_support {metrics['false_support']} loss {metrics['loss']}")
        if better(self.args.select, metrics, st["best_metrics"]):
            st["best_metrics"] = metrics
            st["best"] = {"epoch": row["epoch"], "step": st["step"]}
            if self.args.select != "last":
                self.save_best()
        return metrics

    def save_best(self):
        save_best(os.path.join(self.args.out, "ckpt-best"), self.model, self.tokenizer, self.state["best_metrics"],
                  self.state)

    def check_collapse(self, metrics):
        args = self.args
        if not args.collapse_acc or metrics is None or self.state.get("collapse_checked"):
            return
        if self.state["step"] < args.collapse_step:
            return
        self.state["collapse_checked"] = True
        if metrics["acc"] < args.collapse_acc:
            raise Collapse(f"dev acc {metrics['acc']} < {args.collapse_acc} at step {self.state['step']}")

    def after_step(self, loss):
        args = self.args
        st = self.state
        write_heartbeat(args.heartbeat, st["step"], loss)
        if st["step"] % LOG_EVERY == 0:
            log(f"step {st['step']}/{self.total_steps} epoch {st['epoch'] + 1} loss {loss:.5f} "
                f"lr {self.scheduler.get_last_lr()[0]:.2e}")
        self.watch.check()
        if args.eval_every and st["step"] % args.eval_every == 0:
            self.check_collapse(self.run_eval(epoch_end=False))
        if args.max_steps and st["step"] >= args.max_steps:
            raise Stop("max_steps", 0)
        if st["step"] % args.ckpt_every == 0 or time.time() - self.last_ckpt > args.ckpt_minutes * 60:
            self.checkpoint()

    def train(self):
        args = self.args
        st = self.state
        self.last_ckpt = time.time()
        nan_run = 0
        loss_acc = 0.0
        while st["epoch"] < args.epochs:
            items = self.epoch_items(st["epoch"])
            batches = make_batches(items, args.batch, st["seed"], st["epoch"])
            for mi in range(st["micro"], len(batches)):
                batch = collate(items, batches[mi], self.pad_id, self.device)
                loss = forward_loss(self.model, batch, self.device, self.use_bf16)
                value = loss.detach().float().item()
                if not math.isfinite(value):
                    nan_run += 1
                    # 누적 중이던 기울기를 버리므로 그 몫의 손실 기록도 버린다
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_acc = 0.0
                    log(f"loss {value} at micro {mi} ({nan_run}/{NAN_LIMIT})")
                    if nan_run >= NAN_LIMIT:
                        raise Stop("nan", EXIT_NAN)
                    continue
                nan_run = 0
                (loss / args.accum).backward()
                loss_acc += value / args.accum
                st["micro"] = mi + 1
                if st["micro"] % args.accum != 0 and st["micro"] != len(batches):
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                st["step"] += 1
                self.after_step(loss_acc)
                loss_acc = 0.0
            self.check_collapse(self.run_eval(epoch_end=True))
            st["epoch"] += 1
            st["micro"] = 0
            self.checkpoint()
            write_info(args, self.info("running"))
        if args.select == "last":
            # 마지막 바퀴 가중치를 ckpt-best 로 한 번만 낸다 (바퀴마다 덮어 쓰는 저장 비용을 안 늘린다)
            if st["best"] is None:
                st["best"] = {"epoch": st["epoch"], "step": st["step"]}
            self.save_best()


def run_once(args, seed, lr, restarts):
    """한 판. (stop_reason, exit code, Run)."""
    run = Run(args, seed, lr, restarts)
    log(f"start stage {args.stage} init {args.init} device {args.device} dtype {args.dtype} seed {run.state['seed']} "
        f"lr {run.state['lr']} train {len(run.main_items)} dev {len(run.dev_items)} total_steps {run.total_steps}")
    try:
        run.train()
    except Stop as stop:
        if stop.reason != "nan":
            run.checkpoint()
        write_info(args, run.info(stop.reason))
        log(f"stop {stop.reason}")
        return stop.reason, stop.code, run
    write_info(args, run.info("done"))
    log("done")
    return "done", 0, run


def train_with_restarts(args):
    """무너지면 씨앗 +1 로 최대 collapse_retries 번, 그래도면 lr collapse_lr 로 한 번 (설계 3-2)."""
    plan = [(args.seed + i, args.lr) for i in range(args.collapse_retries + 1)] + [(args.seed, args.collapse_lr)]
    restarts = []
    for seed, lr in plan:
        try:
            return run_once(args, seed, lr, restarts)
        except Collapse as err:
            restarts.append({"seed": seed, "lr": lr, "why": str(err)})
            log(f"collapse: {err} — 다시 시작")
            for name in ("ckpt-last", "ckpt-best"):
                path = os.path.join(args.out, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
            args.resume = False
    write_info(args, {"stop_reason": "collapse", "restarts": restarts})
    return "collapse", EXIT_COLLAPSE, None


# ---------------------------------------------------------------- smoke

SMOKE_SUBJECTS = ["고양이", "서버", "버튼", "창고", "기차", "편지", "등대", "시계"]
SMOKE_TRAITS = ["빠르다", "조용하다", "무겁다", "밝다"]


def smoke_rows(n, seed):
    """손으로 만든 일반 문장 쌍. 지지 · 반대 · 무관이 번갈아 나온다."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        subject = rng.choice(SMOKE_SUBJECTS)
        trait = rng.choice(SMOKE_TRAITS)
        premise = f"{subject} 는 {trait}. 오늘도 그렇다."
        label = i % 3
        if label == 0:
            hypothesis = f"{subject} 는 {trait}."
        elif label == 1:
            hypothesis = f"{subject} 는 {trait} 않다."
        else:
            other = SMOKE_SUBJECTS[(SMOKE_SUBJECTS.index(subject) + 3) % len(SMOKE_SUBJECTS)]
            hypothesis = f"{other} 의 색은 파랗다."
        rows.append({"id": f"s{i:03d}", "premise": premise, "hypothesis": hypothesis, "label": label,
                     "src": "smoke", "kind": "same" if label == 0 else ""})
    return rows


def build_smoke_init(path, rows):
    """글자 단위 토크나이저 + 무작위 초기화 2층 roberta 를 path 에 저장한다 (내려받기 없음)."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for r in rows:
        for ch in r["premise"] + r["hypothesis"]:
            if not ch.isspace() and ch not in vocab:
                vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Split(Regex("."), behavior="isolated")])
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    config = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=SMOKE_MAX_LEN + 2, type_vocab_size=1,
                           pad_token_id=0, bos_token_id=2, eos_token_id=3, num_labels=3)
    torch.manual_seed(0)
    RobertaForSequenceClassification(config).save_pretrained(path, safe_serialization=True)
    fast.save_pretrained(path)


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_smoke(args):
    """64쌍 · batch 8 · 1 바퀴(8 스텝). 4 스텝에서 멈췄다가 --resume 으로 끝까지 — 저장·재개·평가·기록을 본다."""
    os.makedirs(args.out, exist_ok=True)
    rows = smoke_rows(SMOKE_ROWS, 1)
    train_path = os.path.join(args.out, "smoke-train.jsonl")
    dev_path = os.path.join(args.out, "smoke-dev.jsonl")
    write_rows(train_path, rows)
    write_rows(dev_path, rows[:32])
    init = os.path.join(args.out, "smoke-init")
    build_smoke_init(init, rows)
    args.device, args.dtype, args.init = "cpu", "fp32", init
    args.train, args.dev, args.train_epoch, args.replay = [train_path], dev_path, [], ""
    args.batch, args.accum, args.epochs, args.max_len, args.lr = 8, 1, 1, SMOKE_MAX_LEN, 1e-3
    args.eval_every, args.ckpt_every, args.collapse_acc = 4, 2, 0
    if not args.heartbeat:
        args.heartbeat = os.path.join(args.out, "heartbeat.json")
    args.max_steps, args.resume = 4, False
    reason, _, _ = run_once(args, args.seed, args.lr, [])
    if reason != "max_steps":
        raise SystemExit(f"SMOKE FAIL: 첫 판 stop_reason {reason}")
    args.max_steps, args.resume = 0, True
    reason, _, run = run_once(args, args.seed, args.lr, [])
    with open(os.path.join(args.out, "train_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    problems = []
    if reason != "done" or info["stop_reason"] != "done":
        problems.append(f"stop_reason {reason}")
    if info["steps"] != 8:
        problems.append(f"steps {info['steps']} != 8")
    for name in ("ckpt-last/state.pt", "ckpt-best/best.json", "ckpt-best/tokenizer.json"):
        if not os.path.exists(os.path.join(args.out, name)):
            problems.append(f"없음 {name}")
    # 감싸개 아래면 JT_HEARTBEAT 가 out 밖(logs/)을 가리킨다
    if not os.path.exists(args.heartbeat):
        problems.append(f"없음 {args.heartbeat}")
    if not info["dev_history"]:
        problems.append("dev_history 비었음")
    if problems:
        raise SystemExit("SMOKE FAIL: " + " · ".join(problems))
    print("SMOKE OK " + json.dumps({"steps": info["steps"], "dev": info["dev_history"][-1]["acc"],
                                    "best": info["best"], "out": args.out}, ensure_ascii=False), flush=True)
    return 0


def main(argv=None):
    global _log_file
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    _log_file = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    if args.smoke:
        torch.set_num_threads(min(args.threads, 4))
        return run_smoke(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda 를 못 잡았다 — 장치 인자(--device /dev/kfd /dev/dri)를 본다")
    if not args.init or not args.train:
        raise SystemExit("--init 과 --train 이 있어야 한다")
    _, code, _ = train_with_restarts(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
