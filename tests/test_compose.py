"""Static checks of compose.yaml and the Dockerfiles (no Docker needed, so CI can run them)."""

import json
import os
import re
import shlex
import subprocess

import pytest
import yaml

from hs_rag_classifier.eval.items import REPO_ROOT

COMPOSE = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]
DOCKERFILES = [REPO_ROOT / "Dockerfile", REPO_ROOT / "docker" / "llama" / "Dockerfile"]


def test_api_publishes_only_on_localhost():
    assert SERVICES["api"]["ports"] == ["127.0.0.1:8000:8000"]
    for name, svc in SERVICES.items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_llama_has_no_ports_and_is_on_the_internal_network_only():
    llama = SERVICES["llama"]
    assert "ports" not in llama and "expose" not in llama
    assert llama["networks"] == ["internal"]
    assert COMPOSE["networks"]["internal"]["internal"] is True
    assert set(SERVICES["api"]["networks"]) == {"internal", "default"}
    assert SERVICES["api"]["depends_on"]["llama"]["condition"] == "service_healthy"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_every_mount_is_read_only_at_the_same_home_path(name):
    vols = SERVICES[name]["volumes"]
    assert vols
    for v in vols:
        assert isinstance(v, str), v  # short syntax only, so the check below sees everything
        src, dst, mode = v.split(":")
        assert src == dst and src.startswith("${HOME}/") and mode == "ro", v
    assert SERVICES[name]["environment"]["HOME"] == "${HOME}"


def test_no_env_file_and_no_username():
    lines = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8").splitlines()
    text = "\n".join(line.split(" #")[0] for line in lines if not line.lstrip().startswith("#"))
    assert "env_file" not in text and all("env_file" not in s for s in SERVICES.values())
    assert "/home/" not in text
    for svc in SERVICES.values():
        assert not any(re.search(r"SECRET|TOKEN|CLIENT_ID|PASSWORD", k) for k in svc["environment"])


def test_llama_command_is_the_frozen_a4_server_command():
    frozen = json.loads((REPO_ROOT / "configs" / "a4_pipeline.json").read_text())["server"][
        "command"
    ]
    binary, *args = shlex.split(frozen)
    assert binary == "~/llama.cpp/build/bin/llama-server"
    want = [a.replace("~/", "${HOME}/", 1) if a.startswith("~/") else a for a in args]
    assert SERVICES["llama"]["command"] == want
    for flag in ("--offline", "--cache-ram", "-np"):
        assert flag in want
    llama_df = (REPO_ROOT / "docker" / "llama" / "Dockerfile").read_text()
    assert 'ENTRYPOINT ["llama-server"]' in llama_df


def test_both_gpu_services_use_the_nvidia_runtime_and_never_restart():
    for svc in SERVICES.values():
        assert svc["runtime"] == "nvidia" and svc["restart"] == "no"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.parent.name or "api")
def test_dockerfile_base_images_are_pinned_by_tag_and_digest(path):
    froms = re.findall(r"^FROM\s+(\S+)", path.read_text(), flags=re.MULTILINE)
    assert froms
    for ref in froms:
        assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", ref), ref
        assert ":latest" not in ref
    text = path.read_text()
    assert "uv:latest" not in text
    if path == REPO_ROOT / "Dockerfile":
        assert "ghcr.io/astral-sh/uv:0.13.1@sha256:" in text
        assert "uv sync --frozen --no-dev --group llm --group ml --group serve" in text


def test_api_dockerfile_no_uv_cache_and_readable_copies():
    text = (REPO_ROOT / "Dockerfile").read_text()
    syncs = [line for line in text.splitlines() if "uv sync" in line and line.startswith("RUN")]
    assert len(syncs) == 2 and all("--no-cache" in line for line in syncs)
    # Every copied path is made readable (NTFS 0700/0600 modes) before the project install.
    copied = []
    for line in text.splitlines():
        if line.startswith("COPY") and "--from" not in line:
            copied += line.split()[1:-1]
    chmod = next(line for line in text.splitlines() if line.startswith("RUN chmod -R a+rX"))
    assert all(f"/app/{p}" in chmod.split() for p in copied), (copied, chmod)
    lines = text.splitlines()
    assert lines.index(chmod) < max(
        i for i, line in enumerate(lines) if line.startswith("RUN uv sync")
    )


def test_llama_cmake_flags_match_the_host_build():
    text = (REPO_ROOT / "docker" / "llama" / "Dockerfile").read_text()
    m = re.search(r"cmake -B build((?:\s+-D\S+|\s+\\)+)", text)
    flags = sorted(f for f in m.group(1).split() if f.startswith("-D"))
    assert flags == ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=120",
                     "-DGGML_CUDA=ON", "-DGGML_CUDA_NCCL=OFF"]  # fmt: skip
    assert "c35b66744f13cb0dcc476af063e112122eee9355" in text


def test_build_script_refuses_dirty_trees_and_checks_the_import():
    text = (REPO_ROOT / "scripts" / "docker_build.sh").read_text()
    assert "git status --porcelain" in text and "exit 1" in text
    assert "--user 1000:1000" in text and "import hs_rag_classifier.service" in text
    code = [x for x in text.splitlines() if not x.lstrip().startswith("#")]
    # The import check is a throwaway container (--rm); nothing else is ever removed.
    run = [x for x in code if "docker run" in x]
    assert len(run) == 1 and "--rm" in run[0] and "--network none" in run[0]
    assert not any(" rm " in x or "rmi" in x or "prune" in x for x in code)


def test_api_mounts_the_shared_hf_blobs_store_read_only():
    # The HF cache's repo folders hold relative links into hub/blobs (run 2 of step 4b stopped at
    # API start-up with them dangling). llama's GGUF repo is self-contained, so only the api.
    blobs = "${HOME}/.cache/huggingface/hub/blobs"
    assert f"{blobs}:{blobs}:ro" in SERVICES["api"]["volumes"]
    assert not any(v.startswith(blobs) for v in SERVICES["llama"]["volumes"])


def test_image_tags_follow_the_fixed_scheme():
    llama = SERVICES["llama"]["image"]
    assert re.fullmatch(
        r"hs-rag-classifier/llama:b11510-\$\{LLAMA_DOCKERFILE_COMMIT:\?[^}]*\}", llama
    )
    assert re.fullmatch(r"hs-rag-classifier/api:\$\{API_TAG:\?[^}]*\}", SERVICES["api"]["image"])
    # The build arg is always HEAD's commit (the image bakes it), whatever tag is run.
    assert SERVICES["api"]["build"]["args"]["GIT_COMMIT"].startswith("${GIT_COMMIT:?")
    env = (REPO_ROOT / "scripts" / "stack_env.sh").read_text()
    assert "LLAMA_DOCKERFILE_COMMIT=$(git log -1 --format=%h -- docker/llama/Dockerfile)" in env
    assert 'LLAMA_IMAGE="hs-rag-classifier/llama:b11510-${LLAMA_DOCKERFILE_COMMIT}"' in env
    assert 'API_IMAGE="hs-rag-classifier/api:${API_TAG}"' in env


def test_build_script_never_overwrites_an_existing_tag():
    text = (REPO_ROOT / "scripts" / "docker_build.sh").read_text()
    assert '. "$(dirname "$0")/stack_env.sh"' in text
    assert "refusing to build $1: $img already exists" in text
    assert 'docker compose build "${to_build[@]}"' in text
    assert "docker tag" not in text and "--pull" not in text


def test_up_script_runs_the_mount_preflight_before_compose_up():
    lines = (REPO_ROOT / "scripts" / "docker_up.sh").read_text().splitlines()
    pre = next(
        i for i, x in enumerate(lines) if "scripts/preflight_mounts.py" in x and "#" not in x
    )
    up = next(i for i, x in enumerate(lines) if x.startswith("docker compose up"))
    assert pre < up and "set -euo pipefail" in lines
    assert "--no-build" in lines[up] and "--pull never" in lines[up]


def test_dockerignore_keeps_secrets_git_and_data_out():
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    for entry in (".git", ".env", ".env.*", ".venv", "data", "models"):
        assert entry in ignored


# Generic /home/<name> paths allowed in tracked files, by file. Everything else is refused, so no
# personal home path (or user name in one) can enter the tree: paths are relative or ~/..., and
# compose passes the real HOME at build and run time.
HOME_PATHS_ALLOWED = {
    ".devcontainer/devcontainer.json": {"vscode"},  # the dev container image's own user
    "Dockerfile": {"app"},  # neutral ARG APP_HOME default; compose passes ${HOME}
    "docker/llama/Dockerfile": {"app"},
}


def test_no_tracked_file_holds_a_personal_home_path():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True,
                           check=True).stdout.decode().split("\0")  # fmt: skip
    pattern = re.compile(rb"/home/([A-Za-z0-9._-]+)")
    found = []
    for name in filter(None, files):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for m in pattern.finditer(path.read_bytes()):
            user = m.group(1).decode()
            if user not in HOME_PATHS_ALLOWED.get(name, set()):
                found.append(f"{name}: /home/{user}")
    assert not found, found


def _stack_env(**env):
    """Source scripts/stack_env.sh in bash with env and return its exports (or the error)."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    script = '. scripts/stack_env.sh && printf "%s\\n" "$GIT_COMMIT" "$API_TAG" "$API_TAG_SOURCE" "$API_IMAGE"'
    base = {k: v for k, v in os.environ.items() if k != "HS_API_TAG"}
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**base, **env},
    )


def test_api_tag_defaults_to_head():
    r = _stack_env()
    assert r.returncode == 0, r.stderr
    commit, tag, source, image = r.stdout.splitlines()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert (tag, source, image) == (commit, "HEAD", f"hs-rag-classifier/api:{commit}")


def test_hs_api_tag_selects_an_existing_image_and_keeps_head_for_builds():
    r = _stack_env(HS_API_TAG="e65b69627eb0be0dcb1e93910332e548147d51c8")
    assert r.returncode == 0, r.stderr
    commit, tag, source, image = r.stdout.splitlines()
    assert tag == "e65b69627eb0be0dcb1e93910332e548147d51c8" and source == "HS_API_TAG override"
    assert image == f"hs-rag-classifier/api:{tag}"
    assert re.fullmatch(r"[0-9a-f]{40}", commit)  # GIT_COMMIT (the build arg) is still HEAD


@pytest.mark.parametrize("bad", ["-rm", "a b", "x;echo", "../x", "a" * 129])
def test_hs_api_tag_must_be_a_valid_tag(bad):
    r = _stack_env(HS_API_TAG=bad)
    assert r.returncode != 0 and "not a valid image tag" in r.stderr


def test_build_script_refuses_while_hs_api_tag_is_set():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    # Refuses right after sourcing stack_env.sh, before the tree check or any docker command.
    r = subprocess.run(
        ["bash", "scripts/docker_build.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HS_API_TAG": "e65b696", "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 1 and "refusing to build: HS_API_TAG is set" in r.stderr
    text = (REPO_ROOT / "scripts" / "docker_build.sh").read_text()
    assert text.index("HS_API_TAG") < text.index("git status --porcelain")


def test_up_script_reports_the_api_tag_source_and_built_commit():
    text = (REPO_ROOT / "scripts" / "docker_up.sh").read_text()
    assert "tag from ${API_TAG_SOURCE}" in text
    assert "hs_rag_classifier.git_commit" in text
    assert "refusing to start: $img does not exist" in text
