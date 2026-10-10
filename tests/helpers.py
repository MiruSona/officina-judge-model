"""시험 공용 — 툴 뿌리와 train/ · measure/ 를 import 길에 넣고, 없는 패키지를 알려 준다."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "train", "measure"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)


def has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)
