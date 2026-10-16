"""use_eval 하위명령 모듈 모음. 모듈마다 register(subparsers) 하나로 하위명령을 더한다.
선택 기능 — 사설 메모 저장소용. 예시 경로는 자기 환경에 맞춘다.
CORE 를 먼저, 이 폴더의 다른 모듈(register 가 있는 것)을 이름 차례로 등록한다.
"""
import importlib
import os

CORE = ("servers", "sample", "labels", "report")
NOT_SITES = ("common", "proxy")


def module_names():
    here = os.path.dirname(os.path.abspath(__file__))
    extra = sorted(name[:-3] for name in os.listdir(here)
                   if name.endswith(".py") and not name.startswith("_") and name[:-3] not in CORE + NOT_SITES)
    return list(CORE) + extra


def register_all(sub):
    for name in module_names():
        module = importlib.import_module("use_sites." + name)
        if hasattr(module, "register"):
            module.register(sub)
