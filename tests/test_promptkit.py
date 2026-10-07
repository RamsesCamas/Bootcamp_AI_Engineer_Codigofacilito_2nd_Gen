from __future__ import annotations

from pathlib import Path

import pytest

from prompting import CATEGORIES, PromptKit, PromptNotFoundError

PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
VARS = {"company": "Acme Operaciones", "ticket": "El ERP no abre desde la mañana."}


def write_prompt(root: Path, folder: str, file_version: int, *, active: bool, **overrides) -> Path:
    data = {
        "name": folder,
        "version": file_version,
        "active": str(active).lower(),
        "description": "prueba",
        "system": "Sistema de prueba",
        "user": "{{ ticket }}",
        **overrides,
    }
    folder_path = root / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            f"name: {data['name']}",
            f"version: {data['version']}",
            f"active: {data['active']}",
            f"description: {data['description']}",
            "model_params:",
            "  temperature: 0",
            f"system: '{data['system']}'",
            f"user: '{data['user']}'",
        ]
    )
    path = folder_path / f"v{file_version}.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    return path


@pytest.fixture
def kit() -> PromptKit:
    return PromptKit(PROMPTS_ROOT)


def test_loads_v1_and_v2(kit):
    assert kit.versions("ticket_classifier") == [1, 2]
    v1 = kit.get("ticket_classifier", 1)
    v2 = kit.get("ticket_classifier", 2)
    assert (v1.version, v2.version) == (1, 2)
    assert v2.model_params == {"temperature": 0, "max_tokens": 10}


def test_get_without_version_returns_active_v2(kit):
    assert kit.get("ticket_classifier").version == 2


def test_get_specific_version_even_if_inactive(kit):
    assert kit.get("ticket_classifier", 1).version == 1


def test_unknown_prompt_or_version(kit):
    with pytest.raises(PromptNotFoundError):
        kit.get("no_existe")
    with pytest.raises(PromptNotFoundError, match="v99"):
        kit.get("ticket_classifier", 99)


def test_two_active_versions_fail(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True)
    write_prompt(tmp_path, "demo", 2, active=True)
    with pytest.raises(ValueError, match="más de una versión activa") as exc_info:
        PromptKit(tmp_path).get("demo")
    assert "v1.yaml" in str(exc_info.value) and "v2.yaml" in str(exc_info.value)


def test_no_active_version_fails(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=False)
    with pytest.raises(PromptNotFoundError, match="ninguna versión activa"):
        PromptKit(tmp_path).get("demo")


def test_name_must_match_folder(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True, name="otro_nombre")
    with pytest.raises(ValueError, match="carpeta"):
        PromptKit(tmp_path).get("demo")


def test_version_must_match_file(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True, version=7)
    with pytest.raises(ValueError, match="v1.yaml"):
        PromptKit(tmp_path).get("demo")


def test_missing_variable_names_it(kit):
    prompt = kit.get("ticket_classifier", 2)
    with pytest.raises(ValueError, match="company"):
        prompt.render(ticket="hola")


def test_render_returns_system_and_user(kit):
    messages = kit.get("ticket_classifier").render(**VARS)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(m["content"] for m in messages)


def test_ticket_never_in_system_message_of_v2(kit):
    system, user = kit.get("ticket_classifier", 2).render(**VARS)
    assert VARS["ticket"] not in system["content"]
    assert user["content"] == f"<ticket>{VARS['ticket']}</ticket>"


def test_v2_lists_all_categories_and_company(kit):
    system, _ = kit.get("ticket_classifier", 2).render(**VARS)
    assert ", ".join(CATEGORIES) in system["content"]
    assert "Acme Operaciones" in system["content"]


def test_examples_and_categories_available_in_template(tmp_path):
    write_prompt(
        tmp_path,
        "demo",
        1,
        active=True,
        system="{{ categories | length }} categorías, {{ examples | length }} ejemplos",
    )
    (tmp_path / "demo" / "examples.jsonl").write_text(
        '{"ticket": "a", "categoria": "falla"}\n{"ticket": "b", "categoria": "otro"}\n',
        encoding="utf-8",
    )
    system, _ = PromptKit(tmp_path).get("demo").render(ticket="x")
    assert system["content"] == f"{len(CATEGORIES)} categorías, 2 ejemplos"


def test_examples_empty_when_file_missing(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True)
    assert PromptKit(tmp_path).get("demo").examples == []
