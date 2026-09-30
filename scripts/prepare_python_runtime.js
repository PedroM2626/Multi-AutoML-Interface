// Builds a self-contained Python runtime (interpreter + the pinned core stack) under
// runtime/ so the packaged desktop app does not need Python installed by the user.
//
//   node scripts/prepare_python_runtime.js [--force]
//   RUNTIME_REQUIREMENTS=requirements-all.txt RUNTIME_PYTHON_VERSION=3.11 \
//     node scripts/prepare_python_runtime.js   # every engine, see requirements-all.in
//
// Requires: node, and a python on PATH (or PYTHON_BIN) that can run `python -m uv`.
const { execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'runtime');
// The installers ship the core stack. requirements-all.txt adds the engines whose pins drag
// numpy/pandas/scikit-learn back (PyCaret, Lale, TPOT) and torch, which multiplies the payload:
// it is there for a local build, not for a GitHub release asset (2 GiB per file).
const REQUIREMENTS = path.join(ROOT, process.env.RUNTIME_REQUIREMENTS || 'requirements.txt');
// numpy 2.5.0 requires Python >=3.12, so the bundled interpreter is 3.12; requirements-all.txt
// needs 3.11 because pycaret 3.3.2 refuses to import on anything newer.
const PYTHON_VERSION = process.env.RUNTIME_PYTHON_VERSION || '3.12';
const IS_WINDOWS = process.platform === 'win32';

function run(cmd, args, opts = {}) {
    return execFileSync(cmd, args, {
        encoding: 'utf8',
        maxBuffer: 64 * 1024 * 1024,
        stdio: ['ignore', 'pipe', 'pipe'],
        ...opts,
    });
}

// uv is used only to fetch the standalone CPython build; packages go in with the
// interpreter's own pip, because uv refuses to modify an install it managed.
let UV = null;

function resolveUv() {
    if (UV) return UV;
    const pythons = [process.env.PYTHON_BIN, 'python', 'python3'].filter(Boolean);
    for (const candidate of pythons) {
        try {
            run(candidate, ['-m', 'uv', '--version']);
            UV = { cmd: candidate, args: ['-m', 'uv'] };
            return UV;
        } catch {
            /* next candidate */
        }
    }
    try {
        run('uv', ['--version']);
        UV = { cmd: 'uv', args: [] };
        return UV;
    } catch {
        throw new Error(
            'uv is required to fetch the standalone CPython build. ' +
            'Install it with: python -m pip install uv (CI does this in the build job).'
        );
    }
}

function uv(args) {
    const resolved = resolveUv();
    return run(resolved.cmd, [...resolved.args, ...args]);
}

// Hash of the pinned dependency set, so a rebuild is skipped when nothing changed.
function requirementsHash() {
    const raw = fs.readFileSync(REQUIREMENTS);
    return crypto.createHash('sha256').update(raw).update(uv(['--version'])).digest('hex');
}

function locateInstalledPython(dir) {
    // uv leaves one versioned install (cpython-3.12.13-<triple>) plus an alias directory
    // (cpython-3.12-<triple>) pointing at it; only the versioned one is the real tree.
    const candidates = fs
        .readdirSync(dir)
        .filter((name) => /^cpython-\d+\.\d+\.\d+-/.test(name))
        .map((name) => path.join(dir, name))
        .filter((candidate) => fs.existsSync(interpreterPath(candidate)))
        .sort();
    if (candidates.length === 0) throw new Error(`uv did not leave a CPython install in ${dir}`);
    if (candidates.length > 1) {
        console.warn(`Multiple CPython installs in ${dir}, using the newest: ${candidates.at(-1)}`);
    }
    return candidates.at(-1);
}

function interpreterPath(base) {
    return path.join(base, IS_WINDOWS ? 'python.exe' : path.join('bin', 'python3'));
}

function ensureBundledInterpreter(base, sourceDir) {
    /**
     * Return a bundled interpreter that can actually be spawned after the staging tree is gone.

     * On Linux and macOS bin/python3 is a symlink, and copying the tree keeps it a link pointing
     * *outside* runtime/ - into the staging directory this function's caller deletes seconds
     * later. existsSync and even `python3 --version` say yes while that directory still exists,
     * and then the first real spawn answers ENOENT, which is why the release build passed on
     * Windows (python.exe is a plain file) and died on the other two platforms. So replace every
     * link under bin/ with a copy of what it points at, and verify the interpreter can reach its
     * own standard library before the source tree disappears.
     */
    const target = interpreterPath(base);
    const usable = () => {
        try {
            run(target, ['-c', 'import sys, ensurepip; print(sys.version.split()[0])']);
            return true;
        } catch {
            return false;
        }
    };

    if (!IS_WINDOWS) materializeLinks(base, sourceDir, ['bin', 'lib']);
    if (usable()) return target;

    for (const dir of [path.join(base, 'bin'), path.join(sourceDir, 'bin')]) {
        if (!fs.existsSync(dir)) continue;
        for (const name of fs.readdirSync(dir).filter((n) => /^python3\.\d+$/.test(n)).sort().reverse()) {
            const candidate = path.join(dir, name);
            if (!fs.statSync(candidate, { throwIfNoEntry: false })?.isFile()) continue;
            fs.mkdirSync(path.join(base, 'bin'), { recursive: true });
            fs.rmSync(target, { force: true });
            fs.copyFileSync(candidate, target);
            fs.chmodSync(target, 0o755);
            if (usable()) {
                console.log(`Replaced the unusable bin/python3 with a copy of ${candidate}.`);
                return target;
            }
        }
    }
    throw new Error(
        `no usable interpreter after copying into ${path.join(base, 'bin')}; found: ` +
        `${(fs.existsSync(path.join(base, 'bin')) && fs.readdirSync(path.join(base, 'bin')).join(', ')) || 'nothing'}`,
    );
}

function materializeLinks(base, sourceDir, subdirs) {
    /**
     * Replace symlinks in the copied tree with real copies of what they point at.

     * python-build-standalone ships bin/python3 -> bin/python3.12 and
     * lib/libpython3.12.so -> lib/libpython3.12.so.1.0, and some of those links are absolute into
     * the staging directory. Copied as links, they resolve while staging still exists and turn
     * into ENOENT the moment it is deleted - which is how the release build died on Linux and
     * macOS (spawn of runtime/bin/python3, then stat of runtime/lib/libpython3.12.so) while
     * Windows, whose tree has no links, passed.
     */
    for (const sub of subdirs) {
        const dir = path.join(base, sub);
        if (!fs.existsSync(dir)) continue;
        for (const name of fs.readdirSync(dir)) {
            const full = path.join(dir, name);
            if (!fs.lstatSync(full, { throwIfNoEntry: false })?.isSymbolicLink()) continue;
            const link = fs.readlinkSync(full);
            const candidates = [
                path.resolve(path.dirname(full), link),
                link.startsWith(base) ? path.join(sourceDir, path.relative(base, link)) : null,
                path.join(sourceDir, sub, name),
            ].filter(Boolean);
            const real = candidates.find((c) => fs.statSync(c, { throwIfNoEntry: false })?.isFile());
            if (!real) continue;
            fs.rmSync(full, { force: true });
            fs.copyFileSync(real, full);
            fs.chmodSync(full, 0o755);
            console.log(`Materialized ${sub}/${name} from ${real}.`);
        }
    }
}

function findExternallyManagedMarkers(root) {
    const found = [];
    const walk = (dir, depth) => {
        if (depth > 3) return;
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isFile() && entry.name === 'EXTERNALLY-MANAGED') found.push(full);
            else if (entry.isDirectory()) walk(full, depth + 1);
        }
    };
    walk(root, 0);
    return found;
}

// xgboost's libxgboost.dylib links @rpath/libomp.dylib, and a standalone CPython has no
// OpenMP runtime of its own. On macOS that means `import xgboost` (and therefore FLAML's
// default boosting search) fails on any machine without Homebrew, so the library is
// copied into the interpreter's lib directory - one of the paths dyld already searches.
function bundleLibomp(interpreterExe) {
    if (process.platform !== 'darwin') return;
    const libDir = path.join(path.dirname(interpreterExe), 'lib');
    const target = path.join(libDir, 'libomp.dylib');
    if (fs.existsSync(target)) return;

    let source = null;
    const prefixes = [];
    try {
        prefixes.push(run('brew', ['--prefix', 'libomp']).trim());
    } catch {
        /* brew is optional; the known prefixes are tried below */
    }
    for (const prefix of ['/opt/homebrew/opt/libomp', '/usr/local/opt/libomp', ...prefixes]) {
        const candidate = path.join(prefix, 'lib', 'libomp.dylib');
        if (fs.existsSync(candidate)) {
            source = candidate;
            break;
        }
    }
    if (!source) {
        throw new Error(
            'libomp.dylib was not found; the bundled macOS runtime needs it for xgboost. ' +
            'Install it with: brew install libomp'
        );
    }
    fs.mkdirSync(libDir, { recursive: true });
    fs.copyFileSync(source, target);
    console.log(`Bundled OpenMP runtime: ${path.relative(OUT_DIR, target)} (from ${source})`);
}

function main() {
    const force = process.argv.includes('--force');
    const digest = requirementsHash();
    const manifestPath = path.join(OUT_DIR, 'runtime-manifest.json');

    if (!force && fs.existsSync(manifestPath)) {
        const existing = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        const exe = path.join(OUT_DIR, existing.interpreter);
        if (existing.requirementsHash === digest && fs.existsSync(exe)) {
            console.log(`runtime/ already up to date (${existing.interpreter}); use --force to rebuild`);
            return;
        }
    }

    fs.rmSync(OUT_DIR, { recursive: true, force: true });
    const stage = fs.mkdtempSync(path.join(os.tmpdir(), 'ma-runtime-'));

    console.log(`Installing standalone CPython ${PYTHON_VERSION}...`);
    uv(['python', 'install', PYTHON_VERSION, '--install-dir', stage, '--no-bin']);

    const installed = locateInstalledPython(stage);
    // dereference: on Linux/macOS the interpreter is a symlink into uv's staging
    // directory, and copying the link left runtime/bin/python3 pointing back at the
    // managed tree - uv then reported "externally managed" for a path outside runtime/.
    fs.cpSync(installed, OUT_DIR, { recursive: true, force: true, dereference: true });
    // Repaired while `installed` is still on disk: the copy can leave bin/python3 dangling, and
    // the source tree is the only place left to get a real binary from.
    const interpreter = path.relative(OUT_DIR, ensureBundledInterpreter(OUT_DIR, installed));
    console.log(`Interpreter: ${interpreter}`);
    fs.rmSync(stage, { recursive: true, force: true });

    // python-build-standalone ships a PEP 668 marker so system package managers leave it
    // alone. This tree is private to the app and nothing else writes to it, so pip is the
    // right installer here (uv refuses outright to touch an install it managed).
    for (const marker of findExternallyManagedMarkers(OUT_DIR)) {
        fs.rmSync(marker);
        console.log(`Removed ${path.relative(OUT_DIR, marker)}`);
    }

    console.log('Installing the pinned core stack into the bundled interpreter...');
    // With the interpreter's own pip, not `uv pip install --system`: uv changed what it accepts
    // as a system Python between releases and now answers "No system Python installation found
    // for path runtime/bin/python3" for this copied tree on Linux and macOS (Windows still
    // passed), so a payload that must be identical on three platforms cannot depend on the
    // build tool's discovery rules. uv is still what fetches the standalone CPython above.
    const interpreterExe = path.join(OUT_DIR, interpreter);
    try {
        run(interpreterExe, ['-m', 'pip', '--version']);
    } catch {
        run(interpreterExe, ['-m', 'ensurepip', '--upgrade', '--default-pip']);
    }
    run(interpreterExe, ['-m', 'pip', 'install', '--quiet', '--no-input', '-r', REQUIREMENTS]);

    bundleLibomp(interpreterExe);

    const probe = run(interpreterExe, [
        '-c',
        'import sys; import streamlit, mlflow, flaml, pandas, numpy, sklearn; print(sys.version.split()[0])',
    ]);
    console.log(`Import check passed (Python ${probe.trim()}).`);

    fs.writeFileSync(
        manifestPath,
        JSON.stringify(
            {
                interpreter,
                python: probe.trim(),
                platform: process.platform,
                arch: process.arch,
                requirementsHash: digest,
                builtAt: new Date().toISOString(),
            },
            null,
            2
        )
    );

    let bytes = 0;
    const walk = (dir) => {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) walk(full);
            else bytes += fs.statSync(full, { throwIfNoEntry: false })?.size ?? fs.lstatSync(full).size;
        }
    };
    walk(OUT_DIR);
    console.log(`runtime/ ready: ${(bytes / 1e6).toFixed(0)} MB uncompressed.`);
}

main();
