"""The audit gate's exception list, checked without reaching PyPI.

`.github/vuln-exceptions.json` is what lets `pip-audit --strict` tolerate advisories it cannot fix
(pillow's, because Streamlit caps `pillow<13`). The file is only trustworthy while two properties
hold: every row still names the pin that is actually in the lock, and the cap that forces a row is
re-checked rather than assumed. scripts/audit_exceptions.py and scripts/pillow_cap_check.py
implement both, and these tests are the negative controls for them - a check that cannot fail is
not a check.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_exceptions import Stale, flags_for, load_entries, pinned_versions  # noqa: E402
from pillow_cap_check import fix_is_blocked, main as cap_check  # noqa: E402

LOCKS = ("requirements.txt", "requirements-all.txt")


def _write_lock(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_every_lock_named_by_a_row_exists():
    for entry in load_entries():
        for lock in entry["locks"]:
            assert Path(lock).is_file(), f"{entry['package']} names a missing lock: {lock}"


def test_rows_have_a_reason_and_a_revisit_condition():
    for entry in load_entries():
        assert len(entry["reason"]) > 40, entry["package"]
        assert len(entry["revisit"]) > 10, entry["package"]


def test_no_advisory_id_is_excused_by_two_rows():
    ids = [vuln_id for entry in load_entries() for vuln_id in entry["ids"]]
    assert len(ids) == len(set(ids)), sorted(set(i for i in ids if ids.count(i) > 1))


def test_the_rows_describe_the_pins_the_locks_actually_carry():
    """Positive control for the staleness check below: if this is wrong, the gate is not running."""
    for lock in LOCKS:
        pins = pinned_versions(lock)
        for entry in load_entries():
            if lock not in entry["locks"]:
                continue
            normalized = entry["package"].lower().replace("_", "-")
            assert pins[normalized] == entry["version"], (lock, normalized, pins[normalized])


def test_a_moved_pin_refuses_to_inherit_its_predecessors_exception(tmp_path):
    """The rot this prevents: pillow goes to 13.0.0 and the eight ignored ids quietly start
    covering whatever is wrong with the new version."""
    text = Path("requirements.txt").read_text(encoding="utf-8").replace("pillow==12.3.0", "pillow==13.0.0")
    lock = _write_lock(tmp_path, "requirements.txt", text)
    with pytest.raises(Stale, match="pillow is pinned 13.0.0"):
        flags_for(lock)


def test_dropping_a_package_refuses_the_row_to_stay(tmp_path):
    text = "\n".join(line for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
                     if not line.startswith("pillow=="))
    lock = _write_lock(tmp_path, "requirements.txt", text)
    with pytest.raises(Stale, match="no longer lists"):
        flags_for(lock)


def test_the_flags_cover_every_id_of_a_matching_row():
    flags = flags_for("requirements.txt")
    pillow = next(entry for entry in load_entries() if entry["package"] == "pillow")
    assert len(flags) == len(pillow["ids"])
    assert all(flag.startswith("--ignore-vuln=") for flag in flags)


@pytest.mark.parametrize(
    ("specifier", "blocked"),
    [
        ("<13,>=7.1.0", True),      # Streamlit today: 13.0.0 unreachable
        ("<14,>=9.2.0", False),     # cap lifted
        (">=7.1.0", False),         # cap removed entirely
        ("<=12.3.0", True),
        ("~=12.0", True),
        ("~=13.0", False),
        ("==12.3.0", True),
        ("<13.1", False),
    ],
)
def test_the_cap_reader_judges_a_specifier_both_ways(specifier, blocked):
    assert fix_is_blocked([specifier]) is blocked, specifier


def test_the_cap_check_fails_loudly_once_the_cap_is_gone(monkeypatch, tmp_path):
    lock = _write_lock(tmp_path, "requirements.txt", Path("requirements.txt").read_text(encoding="utf-8"))
    monkeypatch.setattr("pillow_cap_check.pillow_specifiers", lambda version: ["<14,>=9.2.0"])
    assert cap_check(str(lock)) == 1
    monkeypatch.setattr("pillow_cap_check.pillow_specifiers", lambda version: ["<13,>=7.1.0"])
    assert cap_check(str(lock)) == 0
    # No pillow requirement at all is also actionable: nothing blocks the fix any more.
    monkeypatch.setattr("pillow_cap_check.pillow_specifiers", lambda version: [])
    assert cap_check(str(lock)) == 1


def test_the_exceptions_file_is_machine_readable():
    data = json.loads(Path(".github/vuln-exceptions.json").read_text(encoding="utf-8"))
    assert isinstance(data["exceptions"], list) and data["exceptions"]
    assert isinstance(data["_comment"], list)


def test_a_filtered_copy_still_matches_the_lock_it_came_from(tmp_path):
    """The nightly audits requirements-all.txt minus pywin32 (no Linux distribution exists, and
    pip-audit enumerates a lock by dry-run installing it). The rows describe the lock the user
    installs, so --as has to keep them matching - otherwise the audit loses its tolerances and goes
    red on packages that cannot be fixed here."""
    text = "\n".join(line for line in Path("requirements-all.txt").read_text(encoding="utf-8").splitlines()
                     if not line.startswith("pywin32=="))
    copy = _write_lock(tmp_path, "requirements-all-audited.txt", text)
    # A name no row claims silently earns no tolerance at all, which is how the step would go red on
    # the four pins it is not allowed to move. --as is what keeps the copy honest.
    assert flags_for(copy) == []
    filtered = flags_for(copy, lock_name="requirements-all.txt")
    whole = flags_for("requirements-all.txt")
    assert filtered == whole and len(filtered) == 28


def test_pywin32_is_the_only_pin_the_runner_cannot_install():
    """The drop list in the workflow is one line long, and that is a measured claim: no pin in this
    lock carries an environment marker to make itself platform-scoped, and pywin32 publishes
    win32/win_amd64/win_arm64 wheels only. If a second such pin appears the step needs it, not a
    guess."""
    lines = Path("requirements-all.txt").read_text(encoding="utf-8").splitlines()
    pinned = [line for line in lines if re.match(r"^[A-Za-z0-9._-]+==", line)]
    assert [line for line in pinned if ";" in line] == []
    assert [line for line in pinned if line.startswith("pywin32==")] == ["pywin32==312"]
    assert len(pinned) == len(pinned_versions("requirements-all.txt"))

