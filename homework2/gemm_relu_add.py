from typing import Tuple

from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule import BlockRV

M = 2048
N = 2048
K = 2048


@T.prim_func
def gemm_relu_add(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
    D: T.Buffer((M, N), "float32"),
) -> None:
    matmul = T.alloc_buffer((M, N), "float32", scope="global")
    relu = T.alloc_buffer((M, N), "float32", scope="global")
    # Compute GeMM
    for i, j, k in T.grid(M, N, K):
        with T.block("gemm"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            vk = T.axis.reduce(K, k)
            with T.init():
                matmul[vi, vj] = T.float32(0)
            matmul[vi, vj] += A[vi, vk] * B[vk, vj]
    # Compute ReLU
    for i, j in T.grid(M, N):
        with T.block("relu"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            relu[vi, vj] = T.max(matmul[vi, vj], T.float32(0))
    # Compute add
    for i, j in T.grid(M, N):
        with T.block("add"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            D[vi, vj] = relu[vi, vj] + C[vi, vj]


def manual_schedule() -> tir.Schedule:
    """The function that manually schedules and optimizes
    the GeMM + ReLU + add workload.

    Returns
    -------
    sch : tir.Schedule
        The schedule of the GeMM + ReLU + add workload.

    Note
    ----
    You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    """
    # Create a schedule from the workload.
    sch = tir.Schedule(gemm_relu_add)
    # Define the shared memory tile sizes and register tile sizes.
    tile_x, tile_y, tile_k = 64, 64, 64
    thread_tile_x, thread_tile_y, thread_tile_k = 4, 4, 1

    # Step 1. Shared memory tiling.
    A_shared, B_shared = shared_memory_tiling(sch, tile_x, tile_y, tile_k)
    # Step 2. Register tiling.
    register_tiling(sch, thread_tile_x, thread_tile_y, thread_tile_k)
    # Step 3. Cooperative fetching.
    cooperative_fetching(
        sch, A_shared, B_shared, tile_x // thread_tile_x, tile_y // thread_tile_y
    )
    # Step 4. Write cache.
    write_cache(sch)
    # Step 5. Epilogue fusion.
    epilogue_fusion(sch)

    return sch


def shared_memory_tiling(
    sch: tir.Schedule, tile_x: int, tile_y: int, tile_k: int
) -> Tuple[BlockRV, BlockRV]:
    """The implementation of shared memory tiling.

    Parameters
    ----------
    sch : tir.Schedule
        The schedule instance.

    tile_x : int
        The shared memory tile size along the `M` dimension.

    tile_y : int
        The shared memory tile size along the `N` dimension.

    tile_k : int
        The shared memory tile size along the `K` dimension.

    Returns
    -------
    A_shared : tir.schedule.BlockRV
        The generated shared memory read stage of `A`.
        It is returned for the cooperative fetching in later tasks.

    B_shared : tir.schedule.BlockRV
        The generated shared memory read stage of `B`.
        It is returned for the cooperative fetching in later tasks.

    Note
    ----
    - You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    - We do not return `sch`, because it is in-place updated during scheduling.
    """
    block_gemm = sch.get_block("gemm")
    # Fetch the loops outside the "gemm" block.
    i, j, k = sch.get_loops(block_gemm)

    # Split loop `i` into an outer loop and an inner loop with regard to tile_x.
    # Here `None` in `factors` means the factor of this loop will be
    # automatically inferred.
    i_outer, i_inner = sch.split(i, factors=[None, tile_x])
    # TODO: Split loop `j` into an outer loop and an inner loop with regard to tile_y.
    ...
    # TODO: Split loop `k` into an outer loop and an inner loop with regard to tile_k.
    ...
    # TODO: Reorder loops into order [i_outer, j_outer, k_outer, i_inner, j_inner, k_inner]
    ...
    # TODO: Bind `i_outer` to blockIdx.x.
    ...
    # TODO: Bind `j_outer` to blockIdx.y.
    ...

    # NOTE: by finishing the schedule above, you have already specified
    # the computation region of each thread block.
    # You can use
    # sch.show()
    # to print out the function to get a sense of how it is right now.
    # We recommend you to frequently use `sch.show()` between step
    # for better understanding of this homework and how the optimization
    # works in general.

    # Generate the shared memory read stage for `A`.
    # `A` is the first buffer in block "gemm"'s read region
    # (shown in `T.reads` in the printed out function).
    # So the read buffer index for `A` is 0.
    A_shared = sch.cache_read(block_gemm, read_buffer_index=0, storage_scope="shared")
    # TODO: Move the read stage to the location under loop `k_outer` with `compute_at`.
    # Think about why we move it under `k_outer`?
    ...
    # TODO: Generate the shared memory read stage for `B`, whose read buffer index is 1.
    ...
    # TODO: Move the read stage to the location under loop `k_outer`.
    ...

    return A_shared, B_shared


def register_tiling(
    sch: tir.Schedule,
    thread_tile_x: int,
    thread_tile_y: int,
    thread_tile_k: int,
) -> None:
    """The implementation of register tiling.

    Parameters
    ----------
    sch : tir.Schedule
        The schedule instance.

    thread_tile_x : int
        The register tile size along the `M` dimension.

    thread_tile_y : int
        The register tile size along the `N` dimension.

    thread_tile_k : int
        The register tile size along the `K` dimension.

    Note
    ----
    - You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    - We do not return `sch`, because it is in-place updated during scheduling.
    """
    block_gemm = sch.get_block("gemm")
    # Fetch the last three loops of the "gemm" block,
    # which are exactly the `i_inner`, `j_inner` and `k_inner` you get
    # in the last task.
    i, j, k = sch.get_loops(block_gemm)[-3:]

    """TODO: Your code here"""
    # Hints:
    # - Make sure you understand the commons and differences
    # between shared memory tiling and register tiling.
    # Follow the same pattern and steps as your implementation
    # in shared memory tiling.
    # - Use "local" as the storage scope of `cache_read` to generate
    # local register read stages.

    ...


def cooperative_fetching(
    sch: tir.Schedule,
    A_shared: BlockRV,
    B_shared: BlockRV,
    thread_extent_x: int,
    thread_extent_y: int,
) -> None:
    """The implementation of cooperative fetching.

    Parameters
    ----------
    sch : tir.Schedule
        The schedule instance.

    A_shared : tir.schedule.BlockRV
        The shared memory read stage of `A` generated by shared memory tiling.

    B_shared : tir.schedule.BlockRV
        The shared memory read stage of `B` generated by shared memory tiling.

    thread_extent_x : int
        The number of threads along the `x` dimension in a thread block,
        or equivalently, the value of `blockDim.x` in GPU.

    thread_extent_y : int
        The number of threads along the `y` dimension in a thread block,
        or equivalently, the value of `blockDim.y` in GPU.

    Note
    ----
    - You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    - We do not return `sch`, because it is in-place updated during scheduling.
    """

    def _cooperative_fetching_impl(block: BlockRV):
        # TODO: Fetch the loops of the read stage with `get_loops`.
        # Think about what loops and how many we want to fetch here?
        ...
        # TODO: Fuse these loops into a single loop.
        ...
        # TODO: Split the fused loop into **three** loops.
        #       The inner two loops should have extent `thread_extent_y`
        #       and `thread_extent_x` respectively.
        ...
        # TODO: Bind two loops among to `threadIdx.x` and `threadIdx.y` respectively.
        ...

    _cooperative_fetching_impl(A_shared)
    _cooperative_fetching_impl(B_shared)


def write_cache(sch: tir.Schedule) -> None:
    """The implementation of write cache.

    Parameters
    ----------
    sch : tir.Schedule
        The schedule instance.

    Note
    ----
    - You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    - We do not return `sch`, because it is in-place updated during scheduling.
    """
    block_gemm = sch.get_block("gemm")
    # TODO: Use `sch.get_loops` to find out the location of inserting write cache.
    loop_index = ...
    write_cache_loc = sch.get_loops(block_gemm)[loop_index]

    # TODO: Generate the local register write stage for GeMM, whose write buffer index is 0.
    ...
    # TODO: Move the generated write cache to the proper location with `reverse_compute_at`.
    ...


def epilogue_fusion(sch: tir.Schedule) -> None:
    """The implementation of epilogue_fusion.

    Parameters
    ----------
    sch : tir.Schedule
        The schedule instance.

    Note
    ----
    - You can use `sch.show()` to print out the function after
    scheduling so far at any time.
    - We do not return `sch`, because it is in-place updated during scheduling.
    """
    # TODO: Use `get_block` to retrieve the addition computation and ReLU computation.
    ...
    # TODO: Use `reverse_compute_inline` to fuse addition into ReLU, and fuse ReLU into GeMM.
    ...


if __name__ == "__main__":
    sch = manual_schedule()
    sch.show()
