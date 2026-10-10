"""N1 학습 벤치 — 무작위 초기화 roberta-large 로 학습 스텝 처리량과 메모리를 잰다.

가중치를 받지 않는다. klue/roberta-large 와 같은 크기(24층 · 1024 · heads 16 · vocab 32000)를
RobertaConfig 로 만들어 무작위로 채운다. 같은 배치를 steps 번 반복해 학습한다.

    python -I bench.py --device cuda --dtype bf16 --batch 16 --seq 128 --steps 20

끝에 `RESULT {...}` JSON 한 줄을 찍는다. 감싸개(run/podman-train.sh)가 그 줄과 로그를 남긴다.
bf16 은 autocast 다 — 가중치·옵티마이저는 fp32 로 두고 계산만 bf16 (설계 3-2).
"""
import argparse
import glob
import json
import math
import os
import threading
import time

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

# klue/roberta-large 크기. 무작위 초기화라 몸통 이름만 같은 꼴이다.
LARGE = dict(vocab_size=32000, num_hidden_layers=24, hidden_size=1024,
             num_attention_heads=16, intermediate_size=4096,
             max_position_embeddings=514, type_vocab_size=1, num_labels=3)
WARMUP_STEPS = 2          # 커널 고르기·할당이 섞이는 앞 스텝은 평균에서 뺀다
MIB = 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(description="N1 무작위 초기화 large 학습 벤치")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--threads", type=int, default=16, help="CPU 길 스레드 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heartbeat", default=os.environ.get("JT_HEARTBEAT", ""),
                   help="스텝마다 덮어 쓰는 심장 박동 파일 (감싸개가 60초 끊김을 본다)")
    p.add_argument("--sample-sec", type=float, default=1.0, help="메모리 표본 간격")
    return p.parse_args()


def read_meminfo():
    """(MemAvailable, 스왑 사용) MiB. 컨테이너 안 /proc/meminfo 는 호스트 값이다."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            vals[key] = int(rest.split()[0])  # kB
    return vals["MemAvailable"] // 1024, (vals["SwapTotal"] - vals["SwapFree"]) // 1024


def read_gtt_used():
    """amdgpu GTT 사용량 MiB (모든 카드 합). sysfs 가 안 보이면 None."""
    paths = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")
    if not paths:
        return None
    total = 0
    for path in paths:
        with open(path) as f:
            total += int(f.read().strip())
    return total // MIB


class MemSampler:
    """뒤에서 가용 메모리 최저 · 스왑 최대 · GTT 최대를 모은다 (설계 3-3 의 최소 감시)."""

    def __init__(self, interval):
        self.interval = interval
        self.avail_start, self.swap_start = read_meminfo()
        self.gtt_start = read_gtt_used()
        self.avail_min, self.swap_max, self.gtt_max = self.avail_start, self.swap_start, self.gtt_start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self):
        avail, swap = read_meminfo()
        self.avail_min = min(self.avail_min, avail)
        self.swap_max = max(self.swap_max, swap)
        gtt = read_gtt_used()
        if gtt is not None and self.gtt_max is not None:
            self.gtt_max = max(self.gtt_max, gtt)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.sample()


def write_heartbeat(path, step, loss):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "step": step, "loss": loss}, f)
    os.replace(tmp, path)


def build_model(device):
    torch.manual_seed(0)
    model = RobertaForSequenceClassification(RobertaConfig(**LARGE)).to(device)
    model.train()
    return model


def make_batch(batch, seq, device, seed):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, LARGE["vocab_size"], (batch, seq), generator=gen)
    labels = torch.randint(0, 3, (batch,), generator=gen)
    return ids.to(device), torch.ones_like(ids).to(device), labels.to(device)


def train_step(model, opt, ids, mask, labels, device, use_bf16):
    # bf16 autocast : matmul·attention 만 bf16, 손실·가중치·옵티마이저는 fp32.
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return out.loss.detach().float().item()


def main():
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda 를 못 잡았다 — 장치 인자(--device /dev/kfd /dev/dri)를 본다")

    sampler = MemSampler(args.sample_sec)
    sampler.start()
    write_heartbeat(args.heartbeat, -1, None)

    model = build_model(args.device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    ids, mask, labels = make_batch(args.batch, args.seq, args.device, args.seed)
    use_bf16 = args.dtype == "bf16"
    is_cuda = args.device == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    times, losses = [], []
    for step in range(args.steps):
        t0 = time.time()
        loss = train_step(model, opt, ids, mask, labels, args.device, use_bf16)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(loss)
        write_heartbeat(args.heartbeat, step, loss)
        print(f"step {step} {times[-1]:.3f}s loss {loss:.5f}", flush=True)

    sampler.stop()
    warm = times[WARMUP_STEPS:] if len(times) > WARMUP_STEPS + 2 else times
    step_s = sum(warm) / len(warm)
    gtt_peak_delta = None
    if sampler.gtt_start is not None:
        gtt_peak_delta = sampler.gtt_max - sampler.gtt_start
    result = dict(
        device=args.device, dtype=args.dtype, batch=args.batch, seq=args.seq, steps=args.steps,
        aotriton=os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
        attn=getattr(model.config, "_attn_implementation", None),
        params_m=round(nparam / 1e6, 1),
        step_s=round(step_s, 4), first_step_s=round(times[0], 2),
        pairs_per_min=round(args.batch / step_s * 60, 1),
        loss_finite=all(math.isfinite(x) for x in losses),
        first_loss=round(losses[0], 5), last_loss=round(losses[-1], 5),
        cuda_max_alloc_mb=round(torch.cuda.max_memory_allocated() / MIB) if is_cuda else None,
        cuda_max_reserved_mb=round(torch.cuda.max_memory_reserved() / MIB) if is_cuda else None,
        gtt_start_mb=sampler.gtt_start, gtt_peak_mb=sampler.gtt_max, gtt_peak_delta_mb=gtt_peak_delta,
        avail_start_mb=sampler.avail_start, min_mem_available_mb=sampler.avail_min,
        swap_delta_mb=sampler.swap_max - sampler.swap_start,
        torch=torch.__version__, hip=getattr(torch.version, "hip", None),
        gpu=torch.cuda.get_device_name(0) if is_cuda else None,
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
