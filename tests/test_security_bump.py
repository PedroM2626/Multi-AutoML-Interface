"""The bump bot's arithmetic, without the network.

`scripts/security_bump.py` decides which pin moves, to what, and whether the rewritten lock still
resolves. Each of those decisions has a way to be quietly wrong: a fix version that predates the pin,
a dependency the new release suddenly demands (aiohttp 3.14.4 raised its yarl floor and would have
broken the stack install on its own), or an exception row left behind after the pin it excuses
moved. These tests are the negative controls for those three, with the PyPI calls stubbed.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import security_bump  # noqa: E402
from audit_exceptions import load_entries  # noqa: E402
from security_bump import (  # noqa: E402
    drop_exceptions_for,
    rewrite,
    select,
    with_dependent_floors,
)


def record(package, version, vuln_id, fixes, aliases=()):
    return {"package": package, "version": version, "id": vuln_id,
            "aliases": list(aliases), "fix_versions": list(fixes)}


@pytest.fixture
def pypi(monkeypatch):
    """A stand-in for the two PyPI reads: available releases, and a release's own floors."""
    releases = {}
    floors = {}

    def set_releases(package, versions):
        releases[package] = versions

    def set_floors(package_version, mapping):
        floors[package_version] = mapping

    monkeypatch.setattr(security_bump, "releases_of", lambda package: sorted(
        releases.get(package, []), key=security_bump.version_key))
    monkeypatch.setattr(security_bump, "dependencies_of", lambda package, version: dict(
        floors.get((package, version), {})))
    return set_releases, set_floors


def test_a_tolerated_advisory_is_not_proposed_as_work(pypi):
    set_releases, _ = pypi
    set_releases("pillow", ["12.3.0", "13.0.0"])
    records = [record("pillow", "12.3.0", "CVE-2026-112895", ["13.0.0"],
                      aliases=["GHSA-p4wg-p73j-c528"])]
    actionable, exposed = select(records, load_entries(), "requirements.txt")
    assert actionable == {} and exposed == []


def test_an_untolerated_advisory_becomes_a_bump(pypi):
    set_releases, _ = pypi
    set_releases("aiohttp", ["3.14.3", "3.14.4", "3.14.5"])
    actionable, exposed = select([record("aiohttp", "3.14.3", "GHSA-2g87-pp6c-29x5", ["3.14.4"])], [], "requirements.txt")
    assert actionable == {"aiohttp": "3.14.4"}
    assert exposed == []


def test_one_release_has_to_clear_every_advisory_of_the_package(pypi):
    """Five advisories naming five floors do not get five bumps: the answer is the highest floor,
    or the lower ones stay exposed."""
    set_releases, _ = pypi
    set_releases("uv", ["0.11.25", "0.12.0", "0.13.4"])
    records = [record("uv", "0.11.25", "PYSEC-2026-4321", ["0.13.4"]),
               record("uv", "0.11.25", "PYSEC-2026-9999", ["0.12.0"])]
    actionable, _ = select(records, [], "requirements.txt")
    assert actionable == {"uv": "0.13.4"}
    set_releases("virtualenv", ["21.14.1", "21.14.2", "21.14.4", "21.14.7", "21.14.9", "21.14.12"])
    spread = [record("virtualenv", "21.14.1", f"CVE-2026-11372{i}", [fix])
              for i, fix in enumerate(["21.14.2", "21.14.4", "21.14.2", "21.14.7", "21.14.9"], start=1)]
    actionable, _ = select(spread, [], "requirements-all.txt")
    assert actionable == {"virtualenv": "21.14.9"}


def test_a_fix_that_only_exists_as_a_prerelease_is_not_invented(pypi):
    set_releases, _ = pypi
    set_releases("demo", ["1.0.0", "1.1.0"])
    actionable, exposed = select([record("demo", "1.0.0", "CVE-2026-5", ["9.9.9"])], [], "requirements.txt")
    assert actionable == {} and len(exposed) == 1


def test_an_advisory_without_a_fix_is_reported_not_guessed(pypi):
    set_releases, _ = pypi
    set_releases("nltk", ["3.10.3"])
    actionable, exposed = select([record("nltk", "3.10.3", "GHSA-8mgp-746c-j5xp", [])], [], "requirements.txt")
    assert actionable == {}
    assert [item["id"] for item in exposed] == ["GHSA-8mgp-746c-j5xp"]


def test_a_fix_older_than_the_pin_is_not_a_bump(pypi):
    """The gate would otherwise "bump" a package backwards whenever a record names an odd floor."""
    set_releases, _ = pypi
    set_releases("numpy", ["2.5.0", "2.6.0"])
    actionable, exposed = select([record("numpy", "2.6.0", "CVE-2026-1", ["2.5.0"])], [], "requirements.txt")
    assert actionable == {}
    assert len(exposed) == 1


def test_only_the_pinned_line_changes(tmp_path):
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "# a comment mentioning aiohttp==3.0.0\naiohttp==3.14.3\nyarl==1.24.2 ; python_version>='3.8'\n",
        encoding="utf-8",
    )
    text = rewrite(str(lock), {"aiohttp": "3.14.4"})
    assert text.splitlines()[0] == "# a comment mentioning aiohttp==3.0.0"
    assert text.splitlines()[1] == "aiohttp==3.14.4"
    assert text.splitlines()[2].startswith("yarl==1.24.2")


def test_a_missing_line_is_an_error_not_a_silent_skip(tmp_path):
    lock = tmp_path / "requirements.txt"
    lock.write_text("flaml==2.6.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no `==` line"):
        rewrite(str(lock), {"torch": "2.0.0"})


def test_the_dependent_floor_is_carried_into_the_bump_set(pypi):
    """The failure this exists to prevent: moving aiohttp alone would break the stack install."""
    set_releases, set_floors = pypi
    set_releases("aiohttp", ["3.14.3", "3.14.4"])
    set_releases("yarl", ["1.24.2", "1.25.1"])
    set_floors(("aiohttp", "3.14.4"), {"yarl": "1.25.1", "multidict": "4.5"})
    pins = {"aiohttp": "3.14.3", "yarl": "1.24.2", "multidict": "6.9.1"}
    final, added = with_dependent_floors("requirements.txt", pins, {"aiohttp": "3.14.4"})
    assert final == {"aiohttp": "3.14.4", "yarl": "1.25.1"}
    assert added == {"yarl": "1.25.1"}


def test_a_floor_already_met_does_not_move_anything(pypi):
    set_releases, set_floors = pypi
    set_releases("aiohttp", ["3.14.4"])
    set_releases("yarl", ["1.25.1"])
    set_floors(("aiohttp", "3.14.4"), {"yarl": "1.25.1"})
    final, added = with_dependent_floors("requirements.txt", {"yarl": "1.25.1"}, {"aiohttp": "3.14.4"})
    assert final == {"aiohttp": "3.14.4"} and added == {}


def _exceptions_tree(tmp_path):
    (tmp_path / ".github").mkdir()
    shutil_copy = Path(".github/vuln-exceptions.json").read_text(encoding="utf-8")
    (tmp_path / ".github" / "vuln-exceptions.json").write_text(shutil_copy, encoding="utf-8")
    return tmp_path


def test_moving_a_pin_drops_its_exception_row_for_that_lock_only(tmp_path, monkeypatch, capsys):
    tree = _exceptions_tree(tmp_path)
    monkeypatch.chdir(tree)
    drop_exceptions_for("requirements.txt", {"pillow": "13.0.0"})
    data = json.loads((tree / ".github/vuln-exceptions.json").read_text(encoding="utf-8"))
    pillow = next(entry for entry in data["exceptions"] if entry["package"] == "pillow")
    assert pillow["locks"] == ["requirements-all.txt"]
    assert "requirements.txt: dropped exception rows pillow 12.3.0" in capsys.readouterr().err


def test_a_row_whose_only_lock_moved_is_removed(tmp_path, monkeypatch):
    tree = _exceptions_tree(tmp_path)
    monkeypatch.chdir(tree)
    drop_exceptions_for("requirements-all.txt", {"pillow": "13.0.0", "nltk": "3.11.0"})
    data = json.loads((tree / ".github/vuln-exceptions.json").read_text(encoding="utf-8"))
    packages = [entry["package"] for entry in data["exceptions"]]
    assert "nltk" not in packages
    assert "pillow" in packages  # still excuses requirements.txt
