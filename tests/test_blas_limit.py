"""Scoped host-BLAS thread limit (spec 2026-10-08-blas-thread-scope)."""
import os
import sys

import numpy as np
import pytest

import gauNEGF.config as config
from gauNEGF import utils

threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_threads():
    return [p["num_threads"] for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"]


@pytest.mark.parametrize("ncpu,cap", [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1), (8, 2), (16, 4),
                                      (40, 10), (64, 16), (96, 16), (192, 16)])
def test_cap_rule(ncpu, cap):
    assert utils.blas_thread_cap(ncpu) == cap


def test_cap_override_is_clipped(monkeypatch):
    monkeypatch.setattr(config, "HOST_BLAS_THREADS", 32)
    assert utils.blas_thread_cap(8) == 8
    assert utils.blas_thread_cap(64) == 32


def test_schedulable_cpus_without_affinity(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert utils.schedulable_cpus() == (os.cpu_count() or 1)


def test_limit_caps_and_restores():
    before = _blas_threads()
    want = utils.blas_thread_cap()
    with utils.blas_limit():
        inside = _blas_threads()
    assert all(n == want for n in inside), (want, inside)
    assert _blas_threads() == before


def test_limit_nested_restores_outer():
    with utils.blas_limit(2):
        with utils.blas_limit(1):
            assert all(n == 1 for n in _blas_threads())
        assert all(n == 2 for n in _blas_threads())


def test_decorator_applies_limit_and_keeps_name():
    want = utils.blas_thread_cap()

    @utils.with_blas_limit
    def probe(x):
        return x, _blas_threads()

    x, inside = probe(3)
    assert x == 3 and all(n == want for n in inside)
    assert probe.__name__ == "probe" and hasattr(probe, "__wrapped__")


def test_no_threadpoolctl_is_a_noop(monkeypatch):
    before = _blas_threads()                      # module object captured at import, unaffected below
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)   # makes `import threadpoolctl` raise
    monkeypatch.setattr(utils, "_warned_no_tpc", False, raising=False)
    with pytest.warns(RuntimeWarning, match="threadpoolctl not installed"):
        with utils.blas_limit():
            a = np.ones((4, 4)) @ np.ones((4, 4))
    assert a[0, 0] == 4.0
    assert _blas_threads() == before


def test_user_xla_flags_survive_import():
    import subprocess
    code = ("import os; os.environ['XLA_FLAGS']='--xla_cpu_enable_fast_math=false'; "
            "import gauNEGF.config; print(os.environ['XLA_FLAGS'])")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1] == "--xla_cpu_enable_fast_math=false"


def test_both_fock_to_p_are_scoped():
    from gauNEGF.scf import NEGF
    from gauNEGF.scfE import NEGFE
    assert hasattr(NEGF.FockToP, "__wrapped__")
    assert hasattr(NEGFE.FockToP, "__wrapped__")
    assert NEGF.FockToP.__doc__ and NEGFE.FockToP.__doc__


def test_limit_is_inert_until_entered():
    before = _blas_threads()
    obj = utils.blas_limit(1)          # created but not entered: must not touch the pools
    assert _blas_threads() == before
    with obj:
        assert all(n == 1 for n in _blas_threads())
    assert _blas_threads() == before
