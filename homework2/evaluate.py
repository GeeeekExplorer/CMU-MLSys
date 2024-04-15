import argparse

import numpy as np
import tvm
import tvm.testing
from tvm import tir

from gemm_relu_add import K, M, N, manual_schedule
from trace_submission import apply_trace

np.random.seed(0)


def build_sch(sch: tir.Schedule) -> tvm.runtime.Module:
    return tvm.build(sch.mod, target="cuda")


def test_numerical_correctness(sch: tir.Schedule, num_rounds: int = 5):
    f = build_sch(sch)

    for i in range(num_rounds):
        A_np = np.random.uniform(-1, 1, size=(M, K)).astype("float32")
        B_np = np.random.uniform(-1, 1, size=(K, N)).astype("float32")
        C_np = np.random.uniform(-1, 1, size=(M, N)).astype("float32")

        D_std = np.maximum(A_np @ B_np, 0) + C_np

        device = tvm.cuda()
        A_tvm = tvm.nd.array(A_np, device)
        B_tvm = tvm.nd.array(B_np, device)
        C_tvm = tvm.nd.array(C_np, device)
        D_tvm = tvm.nd.array(np.zeros((M, N), dtype="float32"), device)
        f(A_tvm, B_tvm, C_tvm, D_tvm)
        tvm.testing.assert_allclose(D_tvm.numpy(), D_std, rtol=1e-4, atol=1e-4)
        print(f"Passing test round {i}...")
    print(f"Passed all tests.")


def evaluate_execution_time(sch: tir.Schedule):
    f = build_sch(sch)

    device = tvm.cuda()
    A_tvm = tvm.nd.empty((M, K), "float32", device)
    B_tvm = tvm.nd.empty((K, N), "float32", device)
    C_tvm = tvm.nd.empty((M, N), "float32", device)
    D_tvm = tvm.nd.empty((M, N), "float32", device)

    t = f.time_evaluator(f.entry_name, device, number=3, repeat=10, min_repeat_ms=100)(
        A_tvm, B_tvm, C_tvm, D_tvm
    ).mean
    print("Execution time: %.2f ms" % (t * 1e3))


def evaluate_naive_func_execution_time():
    from gemm_relu_add import gemm_relu_add

    sch = tir.Schedule(gemm_relu_add)
    i, j, _ = sch.get_loops("gemm")
    io, ii = sch.split(i, [None, 32])
    jo, ji = sch.split(j, [None, 32])
    sch.bind(io, "blockIdx.x")
    sch.bind(jo, "blockIdx.y")
    sch.bind(ii, "threadIdx.x")
    sch.bind(ji, "threadIdx.y")
    sch.reverse_compute_at("relu", ji)
    sch.reverse_compute_inline("add")
    sch.set_scope("gemm", 0, "local")
    # Uncomment the line below to check the naive function.
    # sch.show()

    f = build_sch(sch)

    device = tvm.cuda()
    A_tvm = tvm.nd.empty((M, K), "float32", device)
    B_tvm = tvm.nd.empty((K, N), "float32", device)
    C_tvm = tvm.nd.empty((M, N), "float32", device)
    D_tvm = tvm.nd.empty((M, N), "float32", device)

    t = f.time_evaluator(f.entry_name, device, number=3, repeat=10, min_repeat_ms=100)(
        A_tvm, B_tvm, C_tvm, D_tvm
    ).mean
    print("Naive function execution time: %.2f ms" % (t * 1e3))


def show_cuda(sch: tir.Schedule):
    f = build_sch(sch)
    print(f.imported_modules[0].get_source())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--test", action="store_true")
    args.add_argument("--evaluate-manual", action="store_true")
    args.add_argument("--evaluate-tuned", action="store_true")
    args.add_argument("--evaluate-naive", action="store_true")
    args.add_argument("--show-cuda", action="store_true")
    parsed = args.parse_args()

    if parsed.test:
        sch = manual_schedule()
        test_numerical_correctness(sch)
    if parsed.evaluate_manual:
        sch = manual_schedule()
        evaluate_execution_time(sch)
    if parsed.evaluate_tuned:
        from gemm_relu_add import gemm_relu_add

        sch = tir.Schedule(gemm_relu_add)
        apply_trace(sch)
        evaluate_execution_time(sch)
    if parsed.evaluate_naive:
        evaluate_naive_func_execution_time()
    if parsed.show_cuda:
        sch = manual_schedule()
        show_cuda(sch)
