"""scripts/preflight_mounts.py: every symlink under a service's mounts resolves inside them."""

import importlib.util
import os

import pytest
import yaml

from hs_rag_classifier.eval.items import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "preflight_mounts", REPO_ROOT / "scripts" / "preflight_mounts.py"
)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)


@pytest.fixture
def hub(tmp_path):
    """An HF-style cache under tmp_path (as HOME): a shared blobs store and nested repos.

    models--A: blobs/X -> ../../blobs/61/real (the shared store), snapshots/s/model -> ../../blobs/X
    models--B: self-contained (blobs/Y real), snapshots/s/f -> ../../blobs/Y
    """
    h = tmp_path / "hub"
    (h / "blobs" / "61").mkdir(parents=True)
    (h / "blobs" / "61" / "real").write_text("w")
    a = h / "models--A"
    (a / "blobs").mkdir(parents=True)
    (a / "snapshots" / "s").mkdir(parents=True)
    os.symlink("../../blobs/61/real", a / "blobs" / "X")
    os.symlink("../../blobs/X", a / "snapshots" / "s" / "model")
    b = h / "models--B"
    (b / "blobs").mkdir(parents=True)
    (b / "snapshots" / "s").mkdir(parents=True)
    (b / "blobs" / "Y").write_text("y")
    os.symlink("../../blobs/Y", b / "snapshots" / "s" / "f")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_text("s")
    return tmp_path


def vol(rel):
    return f"${{HOME}}/{rel}:${{HOME}}/{rel}:ro"


def test_link_resolving_inside_a_mount_passes(hub):
    n, problems = pm.check_service([vol("hub/models--B")], str(hub))
    assert (n, problems) == (1, [])


def test_nested_repo_links_into_the_shared_store_pass_only_with_the_store_mounted(hub):
    n, problems = pm.check_service([vol("hub/models--A"), vol("hub/blobs")], str(hub))
    assert (n, problems) == (2, [])
    n, problems = pm.check_service([vol("hub/models--A")], str(hub))
    assert n == 2 and len(problems) == 2  # run 2's failure: both links dangle in the container
    assert all("outside the mount set" in p and p.count("hub/blobs") for p in problems)


def test_link_outside_any_mount_fails(hub):
    os.symlink(str(hub / "outside" / "secret"), hub / "hub" / "models--B" / "abs")
    os.symlink("../../outside/secret", hub / "hub" / "models--B" / "rel")
    _, problems = pm.check_service([vol("hub/models--B")], str(hub))
    assert len(problems) == 2 and all("outside the mount set" in p for p in problems)


def test_broken_link_fails(hub):
    os.symlink("blobs/missing", hub / "hub" / "models--B" / "dangling")
    _, problems = pm.check_service([vol("hub/models--B")], str(hub))
    assert len(problems) == 1 and "broken" in problems[0] and "dangling" in problems[0]


def test_a_chain_through_an_unmounted_folder_fails_even_if_it_ends_inside(hub):
    # The host resolves it, the container cannot: the middle hop is not mounted.
    os.symlink("../hub/models--B/blobs/Y", hub / "outside" / "hop")
    os.symlink("../../outside/hop", hub / "hub" / "models--B" / "via")
    _, problems = pm.check_service([vol("hub/models--B")], str(hub))
    assert len(problems) == 1 and "outside the mount set" in problems[0]


def test_symlink_loop_fails(hub):
    b = hub / "hub" / "models--B"
    os.symlink("loop2", b / "loop1")
    os.symlink("loop1", b / "loop2")
    _, problems = pm.check_service([vol("hub/models--B")], str(hub))
    assert len(problems) == 2 and all("loop" in p for p in problems)


def test_nested_mounts_are_walked_once_and_link_across_them(hub):
    os.symlink("models--B/blobs/Y", hub / "hub" / "top")
    n, problems = pm.check_service([vol("hub"), vol("hub/models--B")], str(hub))
    assert problems == [] and n == 4  # A's two, B's one, top; B not walked twice


def test_missing_mount_source_fails(hub):
    _, problems = pm.check_service([vol("hub/nope")], str(hub))
    assert problems and "does not exist" in problems[0]


def test_only_home_is_expanded():
    with pytest.raises(pm.ComposeFormatError):
        pm.mounts_for(["${DATA}/x:${DATA}/x:ro"], "/h")
    with pytest.raises(pm.ComposeFormatError):
        pm.mounts_for(["${HOME}/x:/elsewhere:ro"], "/h")


def test_parser_matches_yaml_on_the_real_compose_file():
    text = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    want = {k: v.get("volumes", []) for k, v in yaml.safe_load(text)["services"].items()}
    assert pm.parse_volumes(text) == want


def test_parser_refuses_long_syntax_volumes():
    text = "services:\n  a:\n    volumes:\n      - type: bind\n        source: /x\n"
    with pytest.raises(pm.ComposeFormatError):
        pm.parse_volumes(text)


def test_main_refuses_to_start_and_lists_paths(hub, tmp_path, capsys):
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "name: t\nservices:\n  api:\n    image: x\n    volumes:\n"
        f"      - {vol('hub/models--A')}\n  llama:\n    volumes:\n      - {vol('hub/models--B')}\n"
    )
    assert pm.main(["--compose", str(compose), "--home", str(hub)]) == 1
    err = capsys.readouterr().err
    assert "models--A/blobs/X" in err and "models--A/snapshots/s/model" in err and "refusing" in err
    compose.write_text(compose.read_text().replace(
        f"      - {vol('hub/models--A')}\n", f"      - {vol('hub/models--A')}\n      - {vol('hub/blobs')}\n"
    ))  # fmt: skip
    assert pm.main(["--compose", str(compose), "--home", str(hub)]) == 0
