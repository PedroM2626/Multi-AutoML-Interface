"""Bump the pins an audit gate flags, when a fix actually exists, and prove the result installs.

The nightly audit goes red every time a new advisory lands on a pinned package - twice in the last
three days here - and each time the repair is the same twenty minutes: find the fix version, check
what the new release itself demands (aiohttp 3.14.4 raised its yarl floor and would have broken the
stack install if the lock had moved alone), then re-resolve. This script does that walk and either
prints the plan or writes the lock changes; the workflow that calls it opens a pull request only
when the rewritten lock resolves cleanly, and reports instead when it does not.

Usage:
    python scripts/security_bump.py                     # plan for both locks, no files touched
    python scripts/security_bump.py --apply             # rewrite the locks in this checkout
    python scripts/security_bump.py --json              # machine-readable plan

Exit code is 0 when there is nothing to do, 10 when a bump is available and validated, 1 when a
package is exposed with no fix release to move to, or the rewrite does not resolve.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exceptions import is_tolerated, load_entries, normalize, pinned_versions  # noqa: E402

# Per lock: the pip-audit arguments, and the pins to strip before auditing. pip-audit enumerates a
# requirements file by dry-run installing it, so a lock compiled for another platform has to lose the
# lines that platform alone serves - the same adjustment the nightly's all-engine step makes.
LOCKS = {
    "requirements.txt": ([], ()),
    "requirements-all.txt": (["--no-deps"], ("pywin32",)),
}
NOTHING, BUMPED, EXPOSED = 0, 10, 1


def fetch_json(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def audit(lock, extra_args):
    """The advisories pip-audit reports for a lock, as records with their fix versions."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as handle:
        report_path = Path(handle.name)
    command = [sys.executable, "-m", "pip_audit", "-r", lock, *extra_args, "--strict",
               "-f", "json", "-o", str(report_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    body = json.loads(report_path.read_text(encoding="utf-8")) if report_path.stat().st_size else {}
    report_path.unlink(missing_ok=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pip-audit failed on {lock}: {result.stderr[-400:]}")
    records = []
    for dependency in body.get("dependencies", []):
        for vulnerability in dependency.get("vulns", []):
            records.append({
                "package": normalize(dependency["name"]),
                "version": dependency["version"],
                "id": vulnerability["id"],
                "aliases": vulnerability.get("aliases") or [],
                "fix_versions": vulnerability.get("fix_versions") or [],
            })
    return records


def audible_copy(lock, drop):
    """A temp copy without the pins that only another platform serves, for pip-audit to enumerate."""
    if not drop:
        return lock
    skip = tuple(f"{name}==" for name in drop)
    lines = [line for line in Path(lock).read_text(encoding="utf-8").splitlines()
             if not line.strip().startswith(skip)]
    handle, name = tempfile.mkstemp(prefix=Path(lock).stem + "-", suffix=".txt")
    os.close(handle)
    Path(name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def releases_of(package):
    """Every version PyPI serves for a package, oldest first."""
    data = fetch_json(f"https://pypi.org/pypi/{package}/json")
    versions = [version for version, files in data["releases"].items()
                if files and re.match(r"^\d+(\.\d+)*$", version)]
    return sorted(versions, key=version_key)


def version_key(text):
    return tuple(int(part) for part in text.split("."))


def lowest_at_least(versions, floor):
    for version in versions:
        if version_key(version) >= version_key(floor):
            return version
    return None


def target_version(package, current, fix_versions):
    """The smallest published release that clears every fix floor in the advisory set.

    Each advisory names its own floor, so one release has to clear them all: the answer is the
    highest floor, mapped up to the nearest version that exists. Taking the lowest would "fix" one
    advisory and leave the others - virtualenv 21.14.1 has five, at 21.14.2, 21.14.4, 21.14.7 and
    21.14.9.
    """
    versions = releases_of(package)
    candidate = None
    for fix in fix_versions:
        pick = lowest_at_least(versions, fix)
        if pick is None:
            return None
        if candidate is None or version_key(pick) > version_key(candidate):
            candidate = pick
    if candidate and version_key(candidate) < version_key(current):
        return None
    return candidate


def dependencies_of(package, version):
    """{normalized name: floor} from the PEP 508 `>=` clauses of a release's own metadata."""
    data = fetch_json(f"https://pypi.org/pypi/{package}/{version}/json")
    floors = {}
    for requirement in data["info"].get("requires_dist") or []:
        match = re.match(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", requirement)
        if not match:
            continue
        name, specifier = normalize(match.group(1)), match.group(2).split(";")[0]
        for floor in re.findall(r">=\s*(\d+(?:\.\d+)*)", specifier):
            if name not in floors or version_key(floor) > version_key(floors[name]):
                floors[name] = floor
    return floors


def with_dependent_floors(lock, pins, bumps):
    """Widen a bump set until nothing in it demands a pin the lock no longer satisfies."""
    cache, pending, resolved = {}, dict(bumps), {}
    for _ in range(6):
        grew = False
        for package, version in list(pending.items()):
            cache.setdefault((package, version), dependencies_of(package, version))
            for dependency, floor in cache[(package, version)].items():
                if dependency not in pins or dependency in resolved:
                    continue
                if version_key(pins[dependency]) >= version_key(floor):
                    continue
                moved = lowest_at_least(releases_of(dependency), floor)
                if moved is None:
                    raise RuntimeError(f"{package} {version} needs {dependency}>={floor}, not published")
                pending[dependency] = moved
                resolved[dependency] = moved
                grew = True
        if not grew:
            break
    return pending, resolved


def rewrite(lock, bumps):
    text = Path(lock).read_text(encoding="utf-8")
    for package, version in bumps.items():
        pattern = re.compile(rf"^({re.escape(package)}==)[^\s#;]+", re.IGNORECASE | re.MULTILINE)
        replacement = rf"\g<1>{version}"
        text, count = pattern.subn(replacement, text)
        if count == 0:
            raise RuntimeError(f"{lock}: no `==` line for {package} to rewrite")
    return text


def resolve_subset(lock, proposals):
    """Keep the proposals that resolve, instead of abandoning a fixable pin next to a blocked one.

    The scheduled run on 2026-10-22 showed why this is needed: in the engine lock, pillow could not
    move (Streamlit caps it) and virtualenv could (five advisories, fixes published), and the plan
    was judged only as a pair - so it reported "does not resolve" and no pull request was opened for
    the pin that had a fix. The whole set is tried first because that is the ordinary night; only a
    conflict costs the extra passes, and each one drops a single proposal, so the work is bounded by
    the size of the plan.

    Returns the kept proposals, whether they resolve, and pip's own last word.
    """
    order = sorted(proposals)
    detail = ""

    def attempt(names):
        nonlocal detail
        subset = {name: proposals[name] for name in names}
        ok, message = resolves(lock, rewrite(lock, subset))
        detail = message or detail
        return ok, subset

    ok, subset = attempt(order)
    if ok:
        return subset, True, detail

    print(f"{lock}: the full plan does not resolve, testing {len(order)} proposals apart", file=sys.stderr)
    independent = [name for name in order if attempt([name])[0]]
    remaining = list(independent)
    while remaining:
        ok, subset = attempt(remaining)
        if ok:
            return subset, True, detail
        blocked = [name for name in remaining if name not in subset]
        remaining.remove(blocked[-1] if blocked else remaining[-1])
    return {}, False, detail


def resolves(lock, text):
    """Whether pip can still resolve the rewritten lock - the same check the stack install does."""
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / Path(lock).name
        candidate.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
             "-r", str(candidate)],
            capture_output=True, text=True,
        )
    return result.returncode == 0, (result.stderr or result.stdout)[-400:]


def drop_exceptions_for(lock, bumps):
    """A row excuses one package version. Once that pin moves, the row has to go for that lock or
    audit_exceptions.py stops the gate - which is the point, but the bot should not leave it."""
    path = Path(".github/vuln-exceptions.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    kept, dropped = [], []
    for entry in data["exceptions"]:
        if normalize(entry["package"]) in bumps and lock in entry["locks"]:
            entry["locks"] = [other for other in entry["locks"] if other != lock]
            named = len(entry.get("ids") or [])
            excuse = f"{named} ids" if named else f"cap at {entry['tolerate_advisories_fixing_only_at']}"
            dropped.append(f"{entry['package']} {entry['version']} ({excuse})")
            if not entry["locks"]:
                continue
        kept.append(entry)
    if dropped:
        data["exceptions"] = kept
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"{lock}: dropped exception rows {', '.join(dropped)} - they no longer describe a pin",
              file=sys.stderr)


def select(records, entries, lock):
    """Split audit records into {package: fix version} to propose, and what is exposed with no fix.

    Records an exception row already explains are the gate's settled decisions, not work for this
    script. A record whose fix release does not exist yet is reported rather than guessed at.
    """
    actionable, exposed = {}, []
    for record in records:
        if is_tolerated(record, entries, lock):
            continue
        if not record["fix_versions"]:
            exposed.append(record)
            continue
        target = target_version(record["package"], record["version"], record["fix_versions"])
        if target is None:
            exposed.append({**record, "note": "no published release clears the fix floor"})
            continue
        current = actionable.get(record["package"])
        # A package with several advisories needs the release that clears all of them at once.
        if current is None or version_key(target) > version_key(current):
            actionable[record["package"]] = target
    return actionable, exposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="rewrite the locks in this checkout")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = parser.parse_args()

    plan, exposed, notes = [], [], []
    for lock, (extra, drop) in LOCKS.items():
        if not Path(lock).is_file():
            continue
        pins = pinned_versions(lock)
        entries = load_entries()
        audible = audible_copy(lock, drop)
        try:
            actionable, unfixable = select(audit(audible, extra), entries, lock)
        finally:
            if audible != lock:
                Path(audible).unlink(missing_ok=True)
        exposed += [{**item, "lock": lock} for item in unfixable]

        if not actionable:
            continue
        final, added = with_dependent_floors(lock, pins, actionable)
        for package, version in sorted(final.items()):
            if package in added:
                demanders = ", ".join(p for p, v in final.items() if package in dependencies_of(p, v))
                notes.append(f"{lock}: {package} moved to {version} only because {demanders} demands it")
        kept, ok, detail = resolve_subset(lock, final)
        skipped = {name: version for name, version in final.items() if name not in kept}
        plan.append({"lock": lock, "bumps": kept, "resolves": ok, "detail": detail,
                     "blocked": sorted(skipped),
                     "from": {package: pins[package] for package in final}})

    if args.json:
        print(json.dumps({"plan": plan, "exposed": exposed, "notes": notes}, indent=2))
    else:
        for entry in plan:
            state = "resolves" if entry["resolves"] else f"DOES NOT RESOLVE: {entry['detail']}"
            print(f"{entry['lock']}: " + ", ".join(
                f"{name} {entry['from'][name]} -> {version}" for name, version in entry["bumps"].items())
                + f" [{state}]")
        for note in notes:
            print(f"  note: {note}")
        for item in exposed:
            print(f"exposed with no fix: {item['lock']} {item['package']} {item['version']} "
                  f"{item['id']} (fix_versions: {item['fix_versions'] or 'none published'})")
        if not plan and not exposed:
            print("nothing actionable: every advisory is either fixed already or tolerated by "
                  ".github/vuln-exceptions.json")

    if args.apply:
        for entry in plan:
            if not entry["resolves"]:
                print(f"refusing to write {entry['lock']}: the bump does not resolve", file=sys.stderr)
                return 1
            Path(entry["lock"]).write_text(rewrite(entry["lock"], entry["bumps"]), encoding="utf-8")
            drop_exceptions_for(entry["lock"], entry["bumps"])
            # stdout belongs to --json, so progress goes to stderr.
            print(f"wrote {entry['lock']}", file=sys.stderr)

    if exposed:
        return EXPOSED
    return BUMPED if any(entry["resolves"] for entry in plan) else NOTHING


if __name__ == "__main__":
    sys.exit(main())
