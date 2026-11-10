"""Static checks of compose.yaml and the Dockerfiles (no Docker needed, so CI can run them)."""

import json
import re
import shlex

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
    assert "/home/" not in text and not re.search(r"\bcal\b", text)
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


def test_dockerignore_keeps_secrets_git_and_data_out():
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    for entry in (".git", ".env", ".env.*", ".venv", "data", "models"):
        assert entry in ignored
