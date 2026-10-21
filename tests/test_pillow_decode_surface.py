"""The Pillow decode surface, kept small enough to reason about.

`requirements.txt` cannot hold pillow 13.0.0: Streamlit caps `pillow<13`, and the nightly audit
gate carries eight pillow advisories as documented exceptions because of it. Five of those eight
fire when a TIFF, GIF, PPM, PDF or EPS is *opened*, so the tolerance is only defensible while
nothing in this repository hands an attacker-named file to Pillow. These tests guard that premise:
they fail when someone adds a decode site, widens the formats the saliency map opens, or lets the
image picker advertise a format whose decoder is under advisory.

They parse source instead of importing it, so the lean PR gate runs them without matplotlib.
"""
import ast
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import pytest

# Formats whose decoders carry an open advisory in pillow 12.3.0.
ADVISORY_BEARING_FORMATS = (".tif", ".tiff", ".gif", ".ppm", ".pgm", ".pdf", ".eps")

DECODE_SITE_ALLOWLIST = {"src/xai_utils.py"}


def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _module_sources():
    return [Path("app.py"), Path("run.py"), *sorted(Path("src").rglob("*.py"))]


def _decode_sites():
    """Module sources that call `Image.open`, as posix-style relative paths."""
    sites = []
    for path in _module_sources():
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "Image"
            ):
                sites.append(path.as_posix())
    return sites


def _saliency_extensions():
    for node in _tree("src/xai_utils.py").body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SALIENCY_IMAGE_EXTENSIONS"
            for target in node.targets
        ):
            return [ast.literal_eval(elt) for elt in node.value.elts]
    raise AssertionError("SALIENCY_IMAGE_EXTENSIONS is gone from src/xai_utils.py")


def test_the_shipped_code_has_one_pillow_decode_site():
    """A second decode site is a second look at the eight advisories the gate tolerates."""
    assert set(_decode_sites()) == DECODE_SITE_ALLOWLIST


def test_saliency_only_opens_formats_without_an_advisory():
    assert set(_saliency_extensions()) & set(ADVISORY_BEARING_FORMATS) == set()


def test_the_image_picker_does_not_advertise_an_advisory_bearing_format():
    """`type=[...]` is what the user is told to hand over, and the ZIP extractor keeps whatever the
    member was named, so this widget list is the only place the policy is written down."""
    advertised = []
    for node in ast.walk(_tree("app.py")):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "file_uploader":
            for keyword in node.keywords:
                if keyword.arg == "type" and isinstance(keyword.value, ast.List):
                    advertised.append({ast.literal_eval(item) for item in keyword.value.elts})
    assert advertised, "no typed file_uploader left in app.py"
    for listing in advertised:
        assert listing & set(ADVISORY_BEARING_FORMATS) == set(), listing


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("PIL") is None,
    reason="src.xai_utils imports matplotlib and this reads Pillow's own error; runs in the nightly",
)
def test_a_refused_path_never_reaches_pillow(monkeypatch):
    """The filter has to run before `Image.open`, because app.py takes the path out of a CSV column
    with no extension check of its own. cv2 is stubbed so the allowed branch gets as far as Pillow
    without OpenCV being installed - it is in neither lock."""
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    from src.xai_utils import generate_cv_saliency_map

    def reported(image_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert generate_cv_saliency_map(object(), image_path) is None
        return [str(w.message) for w in caught]

    for hostile in ("scan.tif", "scan.tiff", "render.eps", "flat.ppm", "cover.pdf", "anim.gif"):
        messages = reported(hostile)
        assert any("was not opened" in message for message in messages), (hostile, messages)
        assert not any("No such file" in message for message in messages), (hostile, messages)

    # Positive control: an allowed name is handed to Pillow, which is then free to fail on the
    # missing file. Without this, a guard that refuses everything would pass the loop above.
    messages = reported("photo.png")
    assert any("No such file or directory: 'photo.png'" in message for message in messages), messages

