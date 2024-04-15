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

    ...

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
            max_trials_global=64,  # We try 64 schedules in the search space.
            num_trials_per_iter=32,
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
