# Builds retr_st5 (same-topic neighbours, k=5) on CPU so the GPU training is untouched.
"""Run build_retrieval_data.py for st5 (same-topic, k=5) on the CPU.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import retrieval_pilot_prep as P
_knn = P.knn
def knn32(Q, T, *a, **k):
    return _knn(Q.astype(np.float32), T.astype(np.float32), *a, **k)
P.knn = knn32
import build_retrieval_data as B
sys.argv = ["build_retrieval_data.py", "--mode", "same_topic", "--k", "5", "--tag", "st5"]
B.main()
