"""Turn .github/vuln-exceptions.json into pip-audit flags, and refuse stale excuses.

The audit gate has to tolerate some advisories (pillow 13 is unreachable while Streamlit caps
`pillow<13`), but a list of ignored ids rots silently: pip-audit never complains about an id that
matches nothing, so an exception would outlive the pin it was written for and keep swallowing
whatever advisory landed on the replacement version. This script closes that hole: it fails when a
lock no longer pins the exact version an entry was written for, which forces whoever bumps a
package to delete the entry it made obsolete.

Rows come in two kinds. A *named* row lists advisory ids, for the case where no fix exists to talk
about (nltk, h2o). A *cap* row names the release a third-party cap keeps out of reach and excuses
only advisories whose fix is exactly that release - it has to be checked against a live report
(`--report`), because the point is that nothing reported today survives the upgrade the cap blocks.
An advisory fixed anywhere else stays red on purpose.

Usage: python scripts/audit_exceptions.py requirements.txt [--report report.json]
Prints the --ignore-vuln flags for that lock on stdout, diagnostics on stderr.
"""
import argparse
import json
import re
import sys
from pathlib import Path

EXCEPTIONS = Path(__file__).resolve().parent.parent / ".github" / "vuln-exceptions.json"
ID_SHAPE = re.compile(r"^(GHSA|CVE|PYSEC|BIT)-[0-9A-Za-z-]+$")
CAP_FIELD = "tolerate_advisories_fixing_only_at"


class Stale(RuntimeError):
    pass


class NeedsReport(RuntimeError):
    pass


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_versions(lock_path):
    """{normalized name: version} for every `name==version` line in a requirements lock."""
    pins = {}
    for line in Path(lock_path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        pins[normalize(name)] = version.split(";")[0].strip()
    return pins


def load_entries(path=EXCEPTIONS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("exceptions")
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no 'exceptions' list")
    for entry in entries:
        for field in ("package", "version", "locks", "reason", "revisit"):
            if not entry.get(field):
                raise ValueError(f"{entry.get('package', '?')}: missing {field}")
        ids = entry.get("ids") or []
        if not ids and not entry.get(CAP_FIELD):
            raise ValueError(f"{entry['package']}: a row needs ids or a {CAP_FIELD}")
        for vuln_id in ids:
            if not ID_SHAPE.match(vuln_id):
                raise ValueError(f"{entry['package']}: {vuln_id} is not a GHSA/CVE/PYSEC id")
    return entries


def rows_for(entries, lock):
    return [entry for entry in entries if lock in entry["locks"]]


def as_versions(value):
    """A cap row may name one blocked release or several: dropping pkg_resources blocked 83.0.0 and
    every release after it, so a fix at 84.1.0 is just as unreachable as one at 83.0.0."""
    if value is None:
        return set()
    return {value} if isinstance(value, str) else set(value)


def is_tolerated(record, entries, lock):
    """Whether a row already explains one advisory reported for a lock."""
    for entry in rows_for(entries, lock):
        if normalize(entry["package"]) != record["package"]:
            continue
        if entry["version"] != record["version"]:
            continue
        if set(entry.get("ids") or []) & set([record["id"], *record["aliases"]]):
            return True
        blocked = entry.get(CAP_FIELD)
        if blocked and record["fix_versions"] and set(record["fix_versions"]) <= as_versions(blocked):
            return True
    return False


def read_report(path):
    """pip-audit's own JSON output, as records shaped like the ones this gate reasons about."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
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


def flags_for(lock_path, entries=None, lock_name=None, records=None):
    """The --ignore-vuln arguments for one lock, or a Stale exception explaining the refusal.

    `lock_name` names the lock the rows apply to when the file being audited is a filtered copy of
    it - the engine lock is audited without its Windows-only pins on a Linux runner, and the rows
    still describe the lock the user installs.
    """
    entries = load_entries() if entries is None else entries
    lock = lock_name or Path(lock_path).name
    pins = pinned_versions(lock_path)
    flags, applied, auto = [], [], []
    for entry in rows_for(entries, lock):
        package = normalize(entry["package"])
        if package not in pins:
            raise Stale(f"{lock}: {entry.get('ids', [package])[0]} excuses {package}, which this lock no longer lists")
        if pins[package] != entry["version"]:
            raise Stale(
                f"{lock}: {entry['package']} is pinned {pins[package]}, not {entry['version']} - "
                f"the exception is stale, delete it and see what the gate now reports"
            )
        named = list(entry.get("ids") or [])
        blocked = entry.get(CAP_FIELD)
        counted = len(named)
        if named:
            flags += [f"--ignore-vuln={vuln_id}" for vuln_id in named]
        if blocked:
            if records is None:
                raise NeedsReport(
                    f"{lock}: the {entry['package']} row excuses advisories by cap, so it has to be "
                    f"checked against the audit report - pass --report <pip-audit json>"
                )
            blocked_versions = as_versions(blocked)
            capped = [
                record for record in records
                if record["package"] == package and record["version"] == entry["version"]
                and record["fix_versions"]
                and set(record["fix_versions"]) <= blocked_versions
            ]
            for record in capped:
                flags += [f"--ignore-vuln={vuln_id}" for vuln_id in [record["id"], *record["aliases"]]]
            counted += len(capped)
            auto.append(f"{entry['package']}: {len(capped)} advisories fixed only in "
                        f"{'/'.join(sorted(blocked_versions))}")
        applied.append(f"{package} {pins[package]} ({counted} ids)")
    summary = ", ".join(applied) if applied else "nothing"
    if auto:
        summary += " | by cap - " + "; ".join(auto)
    print(f"tolerating for {lock}: {summary}", file=sys.stderr)
    return flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", help="the requirements file being audited")
    parser.add_argument("--as", dest="lock_name",
                        help="name of the lock the rows apply to, when auditing a filtered copy")
    parser.add_argument("--report", help="pip-audit JSON report, required by cap rows")
    args = parser.parse_args()
    try:
        records = read_report(args.report) if args.report else None
        print(" ".join(flags_for(args.lock, lock_name=args.lock_name, records=records)))
    except (Stale, NeedsReport, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"audit_exceptions: {error}", file=sys.stderr)
        sys.exit(1)
