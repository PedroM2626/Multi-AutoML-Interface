"""Turn .github/vuln-exceptions.json into pip-audit flags, and refuse stale excuses.

The audit gate has to tolerate some advisories (pillow 13 is unreachable while Streamlit caps
`pillow<13`), but a list of ignored ids rots silently: pip-audit never complains about an id that
matches nothing, so an exception would outlive the pin it was written for and keep swallowing
whatever advisory landed on the replacement version. This script closes that hole: it fails when a
lock no longer pins the exact version an entry excuses, which forces whoever bumps a package to
delete the entry it made obsolete.

Usage: python scripts/audit_exceptions.py requirements.txt
Prints the --ignore-vuln flags for that lock on stdout, diagnostics on stderr.
"""
import json
import re
import sys
from pathlib import Path

EXCEPTIONS = Path(__file__).resolve().parent.parent / ".github" / "vuln-exceptions.json"
REQUIRED_FIELDS = ("package", "version", "locks", "ids", "reason", "revisit")
ID_SHAPE = re.compile(r"^(GHSA|CVE|PYSEC|BIT)-[0-9A-Za-z-]+$")


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
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            raise ValueError(f"{entry.get('package', '?')}: missing {', '.join(missing)}")
        for vuln_id in entry["ids"]:
            if not ID_SHAPE.match(vuln_id):
                raise ValueError(f"{entry['package']}: {vuln_id} is not a GHSA/CVE/PYSEC id")
    return entries


def flags_for(lock_path, entries=None):
    """The --ignore-vuln arguments for one lock, or a Stale exception explaining the refusal."""
    entries = load_entries() if entries is None else entries
    lock = Path(lock_path).name
    pins = pinned_versions(lock_path)
    flags, applied = [], []
    for entry in entries:
        if lock not in entry["locks"]:
            continue
        package = normalize(entry["package"])
        if package not in pins:
            raise Stale(f"{lock}: {entry['ids'][0]} excuses {package}, which this lock no longer lists")
        if pins[package] != entry["version"]:
            raise Stale(
                f"{lock}: {entry['package']} is pinned {pins[package]}, not {entry['version']} - "
                f"the exception is stale, delete it and see what the gate now reports"
            )
        flags += [f"--ignore-vuln={vuln_id}" for vuln_id in entry["ids"]]
        applied.append(f"{package} {pins[package]} ({len(entry['ids'])} ids)")
    print(f"tolerating for {lock}: {', '.join(applied) if applied else 'nothing'}", file=sys.stderr)
    return flags


class Stale(RuntimeError):
    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    try:
        print(" ".join(flags_for(sys.argv[1])))
    except (Stale, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"audit_exceptions: {error}", file=sys.stderr)
        sys.exit(1)
