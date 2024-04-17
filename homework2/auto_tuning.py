import tempfile

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.space_generator import ScheduleFn

from evaluate import test_numerical_correctness
from gemm_relu_add import gemm_relu_add


def auto_tuning_schedule(sch: tir.Schedule) -> tir.Schedule:
    """The function that defines the schedule space for automatic tuning.

    Parameters
    ----------
    sch : tir.Schedule
        An empty schedule of the GeMM + ReLU + add workload.

    Returns
    -------
    sch : tir.Schedule
        The updated schedule of the GeMM + ReLU + add workload.
    """

    """TODO: Your code here"""
    # NOTE: You may need to set argument `preserve_unit_loops=True`
    # in `compute_at` and `reverse_compute_at` to make it work
    # with auto tuning.

    b0 = sch.get_block(name="gemm", func_name="main")
    l1, l2, l3 = sch.get_loops(block=b0)
    _, tile_x = sch.sample_perfect_tile(l1, 2)
    _, tile_y = sch.sample_perfect_tile(l2, 2)
    _, tile_k = sch.sample_perfect_tile(l3, 2)
    l4, l5 = sch.split(loop=l1, factors=[None, tile_x], preserve_unit_iters=True)
    l6, l7 = sch.split(loop=l2, factors=[None, tile_y], preserve_unit_iters=True)
    l8, l9 = sch.split(loop=l3, factors=[None, tile_k], preserve_unit_iters=True)
    sch.reorder(l4, l6, l8, l5, l7, l9)
    sch.bind(loop=l4, thread_axis="blockIdx.x")
    sch.bind(loop=l6, thread_axis="blockIdx.y")
    b10 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b10, loop=l8, preserve_unit_loops=True, index=-1)
    b11 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b11, loop=l8, preserve_unit_loops=True, index=-1)
    b12 = sch.get_block(name="gemm", func_name="main")
    l13, l14, l15, l16, l17, l18 = sch.get_loops(block=b12)
    thread_extent_x, thread_tile_x = sch.sample_perfect_tile(l16, 2)
    thread_extent_y, thread_tile_y = sch.sample_perfect_tile(l17, 2)
    _, thread_tile_k = sch.sample_perfect_tile(l18, 2)
    l19, l20 = sch.split(loop=l16, factors=[None, thread_tile_x], preserve_unit_iters=True)
    l21, l22 = sch.split(loop=l17, factors=[None, thread_tile_y], preserve_unit_iters=True)
    l23, l24 = sch.split(loop=l18, factors=[None, thread_tile_k], preserve_unit_iters=True)
    sch.reorder(l19, l21, l23, l20, l22, l24)
    sch.bind(loop=l19, thread_axis="threadIdx.x")
    sch.bind(loop=l21, thread_axis="threadIdx.y")
    b25 = sch.cache_read(block=b12, read_buffer_index=0, storage_scope="local")
    sch.compute_at(block=b25, loop=l23, preserve_unit_loops=True, index=-1)
    b26 = sch.cache_read(block=b12, read_buffer_index=1, storage_scope="local")
    sch.compute_at(block=b26, loop=l23, preserve_unit_loops=True, index=-1)
    l27, l28, l29, l30, l31 = sch.get_loops(block=b10)
    l32 = sch.fuse(l30, l31, preserve_unit_iters=True)
    l33, l34, l35 = sch.split(loop=l32, factors=[None, thread_extent_x, thread_extent_y], preserve_unit_iters=True)
    sch.bind(loop=l34, thread_axis="threadIdx.x")
    sch.bind(loop=l35, thread_axis="threadIdx.y")
    l36, l37, l38, l39, l40 = sch.get_loops(block=b11)
    l41 = sch.fuse(l39, l40, preserve_unit_iters=True)
    l42, l43, l44 = sch.split(loop=l41, factors=[None, thread_extent_x, thread_extent_y], preserve_unit_iters=True)
    sch.bind(loop=l43, thread_axis="threadIdx.x")
    sch.bind(loop=l44, thread_axis="threadIdx.y")
    b45 = sch.get_block(name="gemm", func_name="main")
    l46, l47, l48, l49, l50, l51, l52, l53, l54 = sch.get_loops(block=b45)
    b55 = sch.cache_write(block=b45, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b55, loop=l50, preserve_unit_loops=True, index=-1)
    b56 = sch.get_block(name="relu", func_name="main")
    b57 = sch.get_block(name="add", func_name="main")
    sch.reverse_compute_inline(block=b56)
    sch.reverse_compute_inline(block=b57)

    return sch


def auto_tune():
    with tempfile.TemporaryDirectory() as work_dir:
        target = tvm.target.Target(
            {
                "kind": "cuda",
                "max_shared_memory_per_block": 49152,
                "max_threads_per_block": 1024,
                "thread_warp_size": 32,
            }
        )
        # Tune the workload and record the evaluated schedules into the database.
        database = ms.tir_integration.tune_tir(
            mod=gemm_relu_add,
            target=target,
            work_dir=work_dir,
            max_trials_global=256,  # We try 64 schedules in the search space.
            num_trials_per_iter=64,
            space=ScheduleFn(sch_fn=auto_tuning_schedule),
        )
        # Retrieve the best performant schedule from the database.
        sch = ms.tir_integration.compile_tir(database, gemm_relu_add, target)
        assert sch is not None, "No valid schedule found!"
        # Print out the optimized function and the schedule.
        sch.mod.show()
        sch.trace.show()
        # Test the numerical correctness.
        test_numerical_correctness(sch)


if __name__ == "__main__":
    auto_tune()
