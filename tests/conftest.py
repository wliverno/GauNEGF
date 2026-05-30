"""pytest conftest for the gauNEGF test suite.

Ensures every test starts with cwd at the repo root so tests that open
fixture files via relative paths (e.g. 'tests/AuSOC.bethe', 'Au.bethe')
resolve correctly even if a prior test chdir'ed to a tempdir to run
Gaussian and didn't restore cwd.
"""
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(autouse=True)
def _cwd_at_repo_root():
    os.chdir(REPO)
    yield
    os.chdir(REPO)
