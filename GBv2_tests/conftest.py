import os
import sys
import pytest
from dotenv import load_dotenv
import logging

env_path = os.path.join(os.path.dirname(__file__), "GBv2.env")
load_dotenv(dotenv_path=env_path, override=True)
 
def add_repo_to_syspath(repo_root_env="GBv2_DIR") -> str:
    repo_path = os.environ.get(repo_root_env)
    if not repo_path or not os.path.isdir(repo_path):
        raise RuntimeError(
            f"Set {repo_root_env} in GBv2_tests/GBv2.env to your GravityBench-V2 path; got {repo_path!r}."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path

def pytest_addoption(parser):
    parser.addoption("--strict-integration", action="store_true",
                     help="Fail (instead of skip) if integration CSVs are missing.")
 
@pytest.fixture(scope="session", autouse=True)
def add_paths():
    add_repo_to_syspath("GBv2_DIR")
    # v1 only needed for parity test; add if present
    if os.environ.get("GBv1_DIR"):
        add_repo_to_syspath("GBv1_DIR")