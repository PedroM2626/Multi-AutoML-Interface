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

from audit_exceptions import (  # noqa: E402
    NeedsReport,
    Stale,
    flags_for,
    is_tolerated,
    load_entries,
    pinned_versions,
)
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
    ids = [vuln_id for entry in load_entries() for vuln_id in (entry.get("ids") or [])]
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


def _report(tmp_path, advisories):
    """A pip-audit JSON report with one advisory per (package, version, id, fix) tuple."""
    dependencies = {}
    for package, version, vuln_id, fix in advisories:
        dependencies.setdefault((package, version), []).append(
            {"id": vuln_id, "aliases": ["GHSA-" + vuln_id[-4:] + "-x"], "fix_versions": [fix] if fix else []})
    body = {"dependencies": [{"name": name, "version": ver, "vulns": vulns}
                             for (name, ver), vulns in dependencies.items()]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def test_a_cap_row_tolerates_only_what_the_blocked_release_fixes(tmp_path):
    """The rule that replaced a hand-listed 16 ids: excused because 13.0.0 is out of reach, so an
    advisory fixed anywhere else is real exposure and must stay red."""
    entries = load_entries()
    fixed_by_cap = {"package": "pillow", "version": "12.3.0", "id": "CVE-2026-999999",
                    "aliases": [], "fix_versions": ["13.0.0"]}
    fixed_elsewhere = {"package": "pillow", "version": "12.3.0", "id": "CVE-2026-888888",
                       "aliases": [], "fix_versions": ["12.4.0"]}
    unfixed = {"package": "pillow", "version": "12.3.0", "id": "CVE-2026-777777",
               "aliases": [], "fix_versions": []}
    assert is_tolerated(fixed_by_cap, entries, "requirements.txt") is True
    assert is_tolerated(fixed_elsewhere, entries, "requirements.txt") is False
    assert is_tolerated(unfixed, entries, "requirements.txt") is False
    assert is_tolerated({"package": "aiohttp", "version": "3.14.4", "id": "CVE-2026-123456",
                         "aliases": [], "fix_versions": ["3.14.5"]}, entries, "requirements.txt") is False


def test_flags_from_a_report_cover_the_cap_advisories_and_only_those(tmp_path):
    report = _report(tmp_path, [
        ("pillow", "12.3.0", "CVE-2026-112895", "13.0.0"),
        ("pillow", "12.3.0", "CVE-2026-112402", "13.0.0"),
        ("pillow", "12.3.0", "CVE-2026-111111", "12.4.0"),
    ])
    flags = flags_for("requirements.txt", records=__import__("audit_exceptions").read_report(report))
    assert "--ignore-vuln=CVE-2026-112895" in flags
    assert "--ignore-vuln=CVE-2026-112402" in flags
    assert not any("111111" in flag for flag in flags)
    assert len(flags) == 4  # two advisories, id plus alias each


def test_a_cap_row_without_the_report_refuses_to_guess(tmp_path):
    with pytest.raises(NeedsReport, match="has to be checked against the audit report"):
        flags_for("requirements.txt")


def test_a_cap_row_can_name_several_unreachable_releases():
    """Dropping pkg_resources blocked setuptools 83.0.0 and every release after it, so an advisory
    fixed at 84.1.0 is as unreachable as one fixed at 83.0.0."""
    entries = load_entries()
    record_84 = {"package": "setuptools", "version": "80.10.2", "id": "GHSA-xmj3-9gf7-h52w",
                 "aliases": [], "fix_versions": ["84.1.0"]}
    record_83 = {"package": "setuptools", "version": "80.10.2", "id": "PYSEC-2026-3447",
                 "aliases": [], "fix_versions": ["83.0.0"]}
    assert is_tolerated(record_84, entries, "requirements-all.txt") is True
    assert is_tolerated(record_83, entries, "requirements-all.txt") is True


def test_a_reachable_alternative_keeps_the_advisory_red():
    """If one branch of the fix is a release the cap does not block, upgrading could help, so the
    exception must not swallow it."""
    entries = load_entries()
    record = {"package": "setuptools", "version": "80.10.2", "id": "CVE-2026-999999",
              "aliases": [], "fix_versions": ["83.0.0", "82.7.0"]}
    assert is_tolerated(record, entries, "requirements-all.txt") is False


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
    report = _report(tmp_path, [
        ("pillow", "12.3.0", "CVE-2026-112895", "13.0.0"),
        ("scikit-learn", "1.4.2", "PYSEC-2024-110", "1.5.0"),
        ("setuptools", "80.10.2", "PYSEC-2026-3447", "83.0.0"),
    ])
    # A name no row claims earns no tolerance at all - fail closed, the second audit goes red on the
    # pins that cannot move. --as is what keeps a filtered copy honest.
    assert flags_for(copy) == []
    # The real lock name does need the report, because pillow's row is defined by a condition.
    with pytest.raises(NeedsReport, match="pillow"):
        flags_for("requirements-all.txt")
    records = __import__("audit_exceptions").read_report(report)
    filtered = flags_for(copy, lock_name="requirements-all.txt", records=records)
    whole = flags_for("requirements-all.txt", records=records)
    assert filtered == whole
    # 3 cap advisories (id + alias) plus the 6 ids of the two rows with no fix published
    assert len(filtered) == 12


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

