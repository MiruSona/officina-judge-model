"""doctor · serve · stop · proxy — 측정 전 점검과 둘째 모델 띄우기 (설계 5절 · 7절). serve 는 pid 를 servers.json 에 적는다.
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
stop 은 servers.json 에 적힌(이 도구가 띄운) 서버만, 그 포트의 /health 이름이 기록과 같을 때 내린다 — 8092 상시 서버는 안 건드린다.
"""
import ctypes
import os
import platform
import shutil
import signal
import subprocess
import sys
import time

from use_sites import common, proxy

SERVE_PY = os.path.join(common.ROOT, "serve.py")
MEM_LOW_GB = 6.0
UP_WAIT_S = 120


class MemoryStatus(ctypes.Structure):
    """윈도 GlobalMemoryStatusEx 의 MEMORYSTATUSEX."""
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_memory_gb():
    """남는 실메모리 GB. 못 재면 None."""
    if sys.platform == "win32":
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.ullAvailPhys / 2 ** 30, 1)
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
    except OSError:
        return None
    return None


def serving_python(given=""):
    """서빙 venv python — 인자 → JudgeModel/venv-serve. 시스템 python 엔 numpy 가 없을 수 있다."""
    if given:
        return given
    tail = os.path.join("Scripts", "python.exe") if sys.platform == "win32" else os.path.join("bin", "python")
    for base in (os.path.join(common.ROOT, "venv-serve"),):
        path = os.path.join(base, tail)
        if os.path.isfile(path):
            return path
    return ""


def cmd_doctor(a):
    protocol, protocol_sha = common.load_protocol(a.protocol)
    alias = a.alias or common.default_alias(a.repo)
    env = {"at": common.now_iso(), "alias": alias, "repo_sha8": common.default_alias(a.repo)[5:],
           "protocol_sha256": protocol_sha, "machine": f"{platform.system()} {platform.release()} · python {platform.python_version()}",
           "free_memory_gb": free_memory_gb(), "problems": []}
    mem_path = shutil.which("mem")
    env["mem_path"] = mem_path or ""
    if not mem_path and not os.environ.get(common.MEM_ENV):
        env["problems"].append("mem 이 PATH 에 없다")
    else:
        _, out, _ = common.run_mem(["version"], check=False)
        env["mem_version"] = out.strip()[:120]
        config = common.mem_json(["judge", "config", "--json"])
        env["llm_toml"] = {"path": config.get("path", ""), "nli_url": config.get("nli_url", ""),
                           "nli_timeout_ms": config.get("nli_timeout_ms"), "key": bool(config.get("key")),
                           **common.repo_thresholds(config)}
        env["judge_dir_files"] = common.judge_dir_files(a.repo)
    expect = proxy.expected_models(protocol)
    env["models"] = {}
    for name, url in common.models_from_arg(a.models, protocol).items():
        h = proxy.health(url)
        want = expect.get(name, {})
        row = {"url": url, "health": h}
        if h is None:
            row["state"] = "안 닿음"
        elif want and (h.get("name") != want["name"] or "".join(h.get("letters", [])) != want["letters"]):
            row["state"] = f"다른 모델이 떠 있다 ({h.get('name')} {''.join(h.get('letters', []))}) — 기대 {want['name']} {want['letters']}"
        else:
            row["state"] = "ok"
        env["models"][name] = row
    common.write_json(common.alias_dir(alias, "env.json"), env)
    print_doctor(env, a)
    return env


def print_doctor(env, a):
    print(f"별명 {env['alias']} · {env['machine']} · 남는 메모리 {env['free_memory_gb']} GB")
    print(f"mem : {env['mem_path'] or '없음'} · {env.get('mem_version', '')}")
    toml = env.get("llm_toml")
    if toml:
        print(f"llm.toml : {toml['path']} · nli_url {toml['nli_url']} · 문턱(repo) nli_sure {toml['nli_sure']}"
              f" · nli_sure_support {toml['nli_sure_support']} · 키 {'있음' if toml['key'] else '없음'}(옮기지 않는다)")
        print(f"판정 기록 폴더 파일 {env['judge_dir_files']} 개 — 측정 전 (뒤 값은 sample.json judge_dir_files_after ·"
              f" 늘어난 몫은 mem judge clean 대상)")
    for name, row in env["models"].items():
        h = row["health"] or {}
        print(f"{name} {row['url']} : {row['state']} {h.get('name', '')} {h.get('model', '')}")
    v100 = env["models"].get("v100")
    if v100 and v100["state"] != "ok":
        port = v100["url"].rsplit(":", 1)[-1]
        if v100["state"] != "안 닿음":
            port = "8095"
            print("  · 그 포트를 다른 모델이 쓰고 있다 — 다른 포트로 띄우고 --models v100=http://127.0.0.1:8095 로 준다")
        model = os.environ.get("JUDGEMODEL_MODEL_DIR") or "<v1.0.0 모델 폴더>"
        print(f"  v1.0.0 띄우기 (약 3GB) : python -I {os.path.join('measure', 'use_eval.py')} serve --model {model} --port {port}")
        print(f"  끝나면 : python -I {os.path.join('measure', 'use_eval.py')} stop --port {port}")
    free = env["free_memory_gb"]
    if free is not None and free < MEM_LOW_GB:
        print(f"  남는 메모리 {free} GB < {MEM_LOW_GB} — 두 모델 동시(ㄱ) 대신 순차(ㄴ)를 권한다")
    for problem in env["problems"]:
        print("문제 : " + problem)


def read_servers():
    path = common.shared_path("servers.json")
    if os.path.isfile(path):
        return common.read_json(path)
    return {}


def cmd_serve(a):
    url = f"http://127.0.0.1:{a.port}"
    h = proxy.health(url)
    if h is not None:
        raise common.Refuse(f"{a.port} 에 이미 {h.get('name')} 가 떠 있다 — 띄우지 않았다")
    model = os.path.abspath(os.path.expanduser(a.model))
    if not os.path.isfile(os.path.join(model, "model.onnx")):
        raise common.Refuse(f"모델이 없다 : {model}")
    python = serving_python(a.python)
    if not python:
        raise common.Refuse("서빙 venv 가 없다 — --python 으로 onnxruntime 이 든 python 을 준다")
    free = free_memory_gb()
    if free is not None and free < MEM_LOW_GB and not a.force:
        raise common.Refuse(f"남는 메모리 {free} GB < {MEM_LOW_GB} — 띄우지 않았다 (--force 로 무시)")
    logs = os.path.join(os.path.expanduser("~"), ".judgemodel", "logs")
    os.makedirs(logs, exist_ok=True)
    cmd = [python, "-I", SERVE_PY, "--model", model, "--port", str(a.port), "--threads", str(a.threads)]
    if a.compat3:
        cmd.append("--compat3")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(os.path.join(logs, f"use1-{a.port}.log"), "ab") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=flags, start_new_session=sys.platform != "win32")
    for _ in range(UP_WAIT_S):
        time.sleep(1)
        if child.poll() is not None:
            raise common.Refuse(f"서버가 바로 죽었다 ({child.returncode}) — ~/.judgemodel/logs/use1-{a.port}.log")
        h = proxy.health(url)
        if h is not None:
            servers = read_servers()
            servers[str(a.port)] = {"pid": child.pid, "model": model, "name": h.get("name"), "at": common.now_iso()}
            common.write_json(common.shared_path("servers.json"), servers)
            print(f"띄웠다 pid {child.pid} : {h.get('name')} {h.get('model')} {url}")
            return
    raise common.Refuse(f"{UP_WAIT_S}초 안에 /health 가 답하지 않았다")


def cmd_stop(a):
    servers = read_servers()
    entry = servers.get(str(a.port))
    if entry is None:
        raise common.Refuse(f"{a.port} 는 이 도구가 띄운 서버가 아니다 — 내리지 않는다")
    h = proxy.health(f"http://127.0.0.1:{a.port}")
    if h is not None and h.get("name") != entry.get("name"):
        raise common.Refuse(f"{a.port} 에 떠 있는 것은 {h.get('name')} 다 (기록 {entry.get('name')}) — 다른 서버라 내리지 않는다")
    try:
        os.kill(entry["pid"], signal.SIGTERM)
        print(f"내렸다 pid {entry['pid']} ({entry.get('name')}) 포트 {a.port}")
    except OSError as e:
        print(f"pid {entry['pid']} 가 이미 없다 ({e.__class__.__name__}) — 기록만 지운다")
    del servers[str(a.port)]
    common.write_json(common.shared_path("servers.json"), servers)


def cmd_proxy(a):
    """대리 서버만 앞에 띄운다 — 측정이 끝난 뒤에도 queries.jsonl 을 이어 쌓을 때 (Ctrl+C 로 끝)."""
    protocol, _ = common.load_protocol(a.protocol)
    p = proxy.Proxy(common.models_from_arg(a.models, protocol), a.reply, common.alias_dir(a.alias, "queries.jsonl"),
                    a.alias, proxy.expected_models(protocol))
    p.place, p.run = a.place, "live"
    p.probe()
    url = p.start(a.port)
    print(f"대리 서버 {url} · 기록 {p.log_path} · reply {a.reply} — llm.toml nli_url 을 여기로 두면 쌓인다")
    print("  llm.toml 의 nli_timeout_ms 를 5000 이상으로 둔다 — 모델 여럿에 차례로 묻느라 느리다")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        p.stop()


def register(sub):
    s = sub.add_parser("doctor", help="mem · llm.toml 문턱 · 서버 /health · 남는 메모리")
    s.add_argument("--repo", default=".")
    s.add_argument("--alias", default="")
    s.add_argument("--models", default="")
    s.add_argument("--protocol", default="")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("serve", help="serve.py 를 포트에 띄운다 (pid 기록)")
    s.add_argument("--model", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--compat3", action="store_true")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--python", default="")
    s.add_argument("--force", action="store_true", help="남는 메모리 검사를 건너뛴다")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("stop", help="이 도구가 띄운 서버만 내린다")
    s.add_argument("--port", type=int, required=True)
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("proxy", help="기록 대리 서버만 앞에 띄운다 (queries.jsonl 이어 쌓기)")
    s.add_argument("--alias", required=True)
    s.add_argument("--models", default="")
    s.add_argument("--reply", default="v110")
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--place", default="live")
    s.add_argument("--protocol", default="")
    s.set_defaults(func=cmd_proxy)
