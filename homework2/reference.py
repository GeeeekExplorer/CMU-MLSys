# fmt: off
from tvm.script import tir as T


@T.prim_func
def reference_gemm_relu_add(
    A: T.Buffer((2048, 2048), "float32"),
    B: T.Buffer((2048, 2048), "float32"),
    C: T.Buffer((2048, 2048), "float32"),
    D: T.Buffer((2048, 2048), "float32"),
):
    A_shared = T.alloc_buffer((2048, 2048), scope="shared")
    B_shared = T.alloc_buffer((2048, 2048), scope="shared")
    A_shared_local = T.alloc_buffer((2048, 2048), scope="local")
    B_shared_local = T.alloc_buffer((2048, 2048), scope="local")
    matmul_local = T.alloc_buffer((2048, 2048), scope="local")
    for i_0 in T.thread_binding(32, "blockIdx.x"):
        for j_0 in T.thread_binding(32, "blockIdx.y"):
            for k_0 in range(32):
                for ax0_ax1_fused_0 in range(16):
                    for ax0_ax1_fused_1 in T.thread_binding(16, "threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(16, "threadIdx.x"):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(2048, i_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 64)
                                v1 = T.axis.spatial(2048, k_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 64)
                                A_shared[v0, v1] = A[v0, v1]
                for ax0_ax1_fused_0 in range(16):
                    for ax0_ax1_fused_1 in T.thread_binding(16, "threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(16, "threadIdx.x"):
                            with T.block("B_shared"):
                                v0 = T.axis.spatial(2048, k_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 64)
                                v1 = T.axis.spatial(2048, j_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 64)
                                B_shared[v0, v1] = B[v0, v1]
                for i_1_0 in T.thread_binding(16, "threadIdx.x"):
                    for j_1_0 in T.thread_binding(16, "threadIdx.y"):
                        for k_1_0 in range(64):
                            for ax0 in range(4):
                                with T.block("A_shared_local"):
                                    v0 = T.axis.spatial(2048, i_0 * 64 + i_1_0 * 4 + ax0)
                                    v1 = T.axis.spatial(2048, k_0 * 64 + k_1_0)
                                    A_shared_local[v0, v1] = A_shared[v0, v1]
                            for ax0 in range(4):
                                with T.block("B_shared_local"):
                                    v0 = T.axis.spatial(2048, k_0 * 64 + k_1_0)
                                    v1 = T.axis.spatial(2048, j_0 * 64 + j_1_0 * 4 + ax0)
                                    B_shared_local[v0, v1] = B_shared[v0, v1]
                            for i_1_1, j_1_1, k_1_1 in T.grid(4, 4, 1):
                                with T.block("gemm"):
                                    vi = T.axis.spatial(2048, i_0 * 64 + i_1_0 * 4 + i_1_1)
                                    vj = T.axis.spatial(2048, j_0 * 64 + j_1_0 * 4 + j_1_1)
                                    vk = T.axis.reduce(2048, k_0 * 64 + k_1_0 + k_1_1)
                                    with T.init():
                                        matmul_local[vi, vj] = T.float32(0)
                                    matmul_local[vi, vj] += A_shared_local[vi, vk] * B_shared_local[vk, vj]
                        for ax0, ax1 in T.grid(4, 4):
                            with T.block("matmul_local"):
                                v0 = T.axis.spatial(2048, i_0 * 64 + i_1_0 * 4 + ax0)
                                v1 = T.axis.spatial(2048, j_0 * 64 + j_1_0 * 4 + ax1)
                                D[v0, v1] = T.max(matmul_local[v0, v1], T.float32(0)) + C[v0, v1]
