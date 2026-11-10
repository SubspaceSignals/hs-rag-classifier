"""git_info(): git when there is a .git and a git binary (the environment is ignored); otherwise
HS_RAG_GIT_COMMIT and HS_RAG_GIT_DIRTY (the API image); neither: GitInfoError."""

import subprocess

import pytest

from hs_rag_classifier.eval import runner as R

SHA = "0123456789abcdef0123456789abcdef01234567"
ENV = {R.ENV_GIT_COMMIT: SHA, R.ENV_GIT_DIRTY: "false"}


def test_git_wins_over_the_environment_in_a_checkout():
    head = subprocess.run(["git", "-C", str(R.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()  # fmt: skip
    got = R.git_info(environ={R.ENV_GIT_COMMIT: SHA, R.ENV_GIT_DIRTY: "true"})
    assert got["commit"] == head and got["commit"] != SHA and isinstance(got["dirty"], bool)
    assert set(got) == {"commit", "dirty"}


@pytest.mark.parametrize("dirty", ["false", "true"])
def test_environment_when_there_is_no_dot_git(tmp_path, dirty):
    got = R.git_info(repo=tmp_path, environ={**ENV, R.ENV_GIT_DIRTY: dirty})
    assert got == {"commit": SHA, "dirty": dirty == "true"}


def test_environment_when_there_is_no_git_binary():
    got = R.git_info(repo=R.REPO_ROOT, environ=ENV, which=lambda name: None)
    assert got == {"commit": SHA, "dirty": False}


@pytest.mark.parametrize("env", [
    {},                                                    # neither source
    {R.ENV_GIT_COMMIT: SHA},                               # dirty flag missing
    {R.ENV_GIT_DIRTY: "false"},                            # commit missing
    {R.ENV_GIT_COMMIT: "38af300", R.ENV_GIT_DIRTY: "false"},  # short hash
    {R.ENV_GIT_COMMIT: SHA, R.ENV_GIT_DIRTY: "0"},         # not "true"/"false"
])  # fmt: skip
def test_neither_source_is_an_error(tmp_path, env):
    with pytest.raises(R.GitInfoError, match="no git provenance: no .git"):
        R.git_info(repo=tmp_path, environ=env)
    with pytest.raises(R.GitInfoError, match="no git binary"):
        R.git_info(repo=R.REPO_ROOT, environ=env, which=lambda name: None)
