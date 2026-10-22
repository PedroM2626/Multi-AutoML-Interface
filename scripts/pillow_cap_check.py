"""Fail the nightly when the pillow exception list stops being necessary.

The audit gate tolerates eight pillow advisories because Streamlit's `pillow<13` cap keeps the
fixed release out of reach. Tolerating them is only honest while that cap holds, and the day
Streamlit lifts it the lock should move to pillow 13 and the entry in
.github/vuln-exceptions.json should disappear - not stay behind as a comfortable silence.

Usage: python scripts/pillow_cap_check.py [requirements.txt]
Exit 0 while the pinned Streamlit still blocks the fix version; 1 when it does not, or when this
cannot be established.
"""
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exceptions import normalize, pinned_versions  # noqa: E402

# The release closing every pillow advisory reported so far; the count grows, this does not need
# editing, because .github/vuln-exceptions.json excuses advisories by this version, not by id.
PILLOW_FIX = (13, 0, 0)


def pillow_specifiers(streamlit_version):
    url = f"https://pypi.org/pypi/streamlit/{streamlit_version}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        info = json.load(response)["info"]
    specifiers = []
    # PEP 508 strings with no space before the specifier are the norm: "pillow<13,>=7.1.0".
    for requirement in info.get("requires_dist") or []:
        match = re.match(r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", requirement)
        if not match:
            continue
        name, specifier = match.group(1), match.group(2)
        if normalize(name) == "pillow":
            specifiers.append(specifier.split(";")[0].strip())
    return specifiers


def clauses(specifier):
    return re.findall(r"(<=|>=|~=|==|!=|<|>)\s*(\d+(?:\.\d+)*)", specifier)


def excludes_fix(operator, version):
    """Whether one specifier clause rules PILLOW_FIX out. A conjunction is blocked if any clause is."""
    parts = tuple(int(number) for number in version.split("."))
    padded = parts + (0,) * (3 - len(parts))
    if operator == "<":
        return PILLOW_FIX >= padded
    if operator == "<=":
        return PILLOW_FIX > padded
    if operator == ">":
        return PILLOW_FIX <= padded
    if operator == ">=":
        return PILLOW_FIX < padded
    if operator == "==":
        return PILLOW_FIX != padded
    if operator == "!=":
        return PILLOW_FIX == padded
    if operator == "~=":
        if len(parts) < 2:
            return PILLOW_FIX[0] != parts[0]
        return PILLOW_FIX[: len(parts) - 1] != parts[: len(parts) - 1] or PILLOW_FIX < padded
    return False


def fix_is_blocked(specifiers):
    return any(excludes_fix(operator, version) for specifier in specifiers for operator, version in clauses(specifier))


def main(lock="requirements.txt"):
    pins = pinned_versions(lock)
    if "pillow" not in pins:
        print(f"pillow_cap_check: {lock} no longer pins pillow - the exception is stale")
        return 1
    if "streamlit" not in pins:
        print(f"pillow_cap_check: {lock} no longer pins streamlit, so nothing caps pillow")
        return 1
    try:
        specifiers = pillow_specifiers(pins["streamlit"])
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as error:
        print(f"pillow_cap_check: inconclusive, could not read streamlit {pins['streamlit']}: {error}")
        return 1
    if not specifiers:
        print(
            f"pillow_cap_check: streamlit {pins['streamlit']} declares no pillow requirement, so "
            f"pillow {pins['pillow']} can move to {'.'.join(map(str, PILLOW_FIX))} - raise the pin "
            f"and delete the pillow entry from .github/vuln-exceptions.json"
        )
        return 1
    if not fix_is_blocked(specifiers):
        print(
            f"pillow_cap_check: streamlit {pins['streamlit']} now allows pillow "
            f"{'.'.join(map(str, PILLOW_FIX))} (specifier: {','.join(specifiers)}) while the lock "
            f"pins {pins['pillow']} - raise the pin and empty the exception list"
        )
        return 1
    print(
        f"pillow_cap_check: streamlit {pins['streamlit']} caps pillow ({','.join(specifiers)}), "
        f"so the tolerated advisories on pillow {pins['pillow']} have no reachable fix"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
