#!/usr/bin/env bash
# Build the MieWorkbench AppImage (GUI + bundled CLI).
#
#   packaging/appimage/build_appimage.sh
#
# Output: $WORK_DIR/MieWorkbench-<version>-x86_64.AppImage
# (default WORK_DIR = <repo>/build/appimage). The version comes from
# mieworkbench/__init__.py. See packaging/appimage/README.md.
#
# Environment knobs (all optional):
#   WORK_DIR        build directory                        (build/appimage)
#   APPIMAGETOOL    path to an appimagetool binary/AppImage (downloaded if unset
#                   and no RUNTIME_FILE is given)
#   RUNTIME_FILE    AppImage type2 runtime ELF; with it (and no APPIMAGETOOL)
#                   the image is assembled with mksquashfs + this runtime
#   SQUASH_COMP     mksquashfs compressor for the fallback path        (zstd)
#   MM_VERSION      micromamba build from conda-forge             (2.10.0-0)
#   PYTHON_SPEC     conda python spec                             (3.12.*)
#   GLIBC_MAX       fail if any ELF needs a newer glibc than this     (2.35)
#   KEEP_CACHE=1    keep downloaded conda caches and the build env
#   SKIP_DEMOS=1    leave demos/ out of the payload
# The script is idempotent: AppDir is rebuilt from scratch on every run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${WORK_DIR:-$REPO/build/appimage}"
MM_VERSION="${MM_VERSION:-2.10.0-0}"
PYTHON_SPEC="${PYTHON_SPEC:-3.12.*}"
GLIBC_MAX="${GLIBC_MAX:-2.35}"
ARCH=x86_64

VERSION="$(sed -n 's/^__version__ *= *"\(.*\)".*/\1/p' \
    "$REPO/mieworkbench/__init__.py" | head -n1)"
[ -n "$VERSION" ] || { echo "cannot read __version__" >&2; exit 1; }
OUT_NAME="MieWorkbench-${VERSION}-${ARCH}.AppImage"

APPDIR="$WORK/AppDir"
USR="$APPDIR/usr"
SHARE="$USR/share/mieworkbench"
TOOLS="$WORK/tools"
MAMBA_ROOT="$WORK/mamba-root"
BUILDENV="$WORK/buildenv"

log() { printf '\n== %s\n' "$*" >&2; }
die() { echo "build_appimage: $*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"
[ -n "$WORK" ] && [ "$WORK" != "/" ] || die "bad WORK_DIR"
mkdir -p "$WORK" "$TOOLS"

# ---------------------------------------------------------------- micromamba
MM="$TOOLS/micromamba"
if [ ! -x "$MM" ] || ! "$MM" --version 2>/dev/null | grep -q "^${MM_VERSION%-*}"; then
    log "fetching micromamba $MM_VERSION"
    tmp="$(mktemp -d "$WORK/mm.XXXXXX")"
    curl -fsSL "https://conda.anaconda.org/conda-forge/linux-64/micromamba-${MM_VERSION}.tar.bz2" \
        -o "$tmp/mm.tar.bz2"
    tar -xjf "$tmp/mm.tar.bz2" -C "$tmp" bin/micromamba
    mv "$tmp/bin/micromamba" "$MM"
    rm -rf "${tmp:?}"
fi
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export MAMBA_NO_BANNER=1
mm() { "$MM" "$@"; }

# ------------------------------------------------------ runtime env (AppDir/usr)
log "creating relocatable python in AppDir/usr"
rm -rf "${APPDIR:?}"
mkdir -p "$APPDIR"
# libgomp: the C engine's OpenMP runtime. xcb-util-* and libxkbcommon: Qt's
# xcb platform plugin needs them and they are not on every desktop.
mm create -y -q -p "$USR" -c conda-forge --override-channels \
    "python=${PYTHON_SPEC}" pip libgomp \
    xcb-util-cursor xcb-util-image xcb-util-keysyms xcb-util-renderutil \
    xcb-util-wm xcb-util libxkbcommon >/dev/null

PY="$USR/bin/python3"
PIPCACHE="$WORK/pip-cache"
# retries/timeout: the Qt and VTK wheels are large; a stalled connection
# is retried instead of hanging the build
PIP_CACHE_DIR="$PIPCACHE" "$PY" -m pip install --disable-pip-version-check -q \
    --retries 10 --timeout 30 \
    --only-binary :all: -r "$HERE/requirements-gui.txt"

# ------------------------------------------------------------ C engine (CPU)
log "building CPU-only C engine (miewb-trace)"
# sysroot_linux-64=2.17 pins the glibc the engine is compiled against
# (the default sysroot tracks a very recent glibc and would raise the
# minimum system requirement).
rm -rf "${BUILDENV:?}"
mm create -y -q -p "$BUILDENV" -c conda-forge --override-channels \
    c-compiler "sysroot_linux-64=2.17" cmake ninja patchelf binutils >/dev/null
rm -rf "${WORK:?}/cengine-build"
mm run -p "$BUILDENV" cmake -G Ninja -S "$REPO/cengine" -B "$WORK/cengine-build" \
    -DCMAKE_BUILD_TYPE=Release -DMIEWB_PORTABLE=ON -DMIEWB_ENABLE_CUDA=OFF \
    >"$WORK/cengine-cmake.log" 2>&1 || { cat "$WORK/cengine-cmake.log" >&2; die "cmake failed"; }
mm run -p "$BUILDENV" ninja -C "$WORK/cengine-build" \
    >"$WORK/cengine-ninja.log" 2>&1 || { tail -50 "$WORK/cengine-ninja.log" >&2; die "ninja failed"; }
install -m 755 "$WORK/cengine-build/miewb-trace" "$USR/bin/miewb-trace"
mm run -p "$BUILDENV" strip --strip-debug "$USR/bin/miewb-trace" || true
mm run -p "$BUILDENV" patchelf --set-rpath '$ORIGIN/../lib' "$USR/bin/miewb-trace"

# Qt's xcb plugin must find the bundled xcb-util libs without an inherited
# LD_LIBRARY_PATH (which would leak into FreeCAD / optics-env children).
SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
XCB="$SP/PySide6/Qt/plugins/platforms/libqxcb.so"
if [ -f "$XCB" ]; then
    rel="$("$PY" -c 'import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))' \
        "$USR/lib" "$(dirname "$XCB")")"
    old="$(mm run -p "$BUILDENV" patchelf --print-rpath "$XCB" || true)"
    mm run -p "$BUILDENV" patchelf --set-rpath "${old:+$old:}\$ORIGIN/$rel" "$XCB"
fi

# --------------------------------------------------------------- PySide6 trim
log "pruning unused Qt modules"
PS="${SP:?}/PySide6"
for m in WebEngineCore WebEngineWidgets WebEngineQuick WebChannel WebSockets \
         WebView Quick QuickWidgets QuickControls2 QuickTemplates2 QuickTest \
         QuickDialogs2 QuickLayouts Qml QmlCore QmlModels QmlWorkerScript \
         3DCore 3DRender 3DInput 3DLogic 3DAnimation 3DExtras 3DQuick \
         Quick3D Quick3DRuntimeRender Quick3DUtils Multimedia MultimediaWidgets \
         SpatialAudio Designer UiTools Help Test Sql Bluetooth Nfc Sensors \
         SerialPort SerialBus Positioning Location RemoteObjects Scxml \
         StateMachine TextToSpeech VirtualKeyboard DataVisualization Graphs \
         GraphsWidgets HttpServer Pdf PdfWidgets ShaderTools NetworkAuth \
         Concurrent Xml; do
    rm -f "${PS:?}"/Qt"$m".* "${PS:?}"/Qt/lib/libQt6"$m".so* 2>/dev/null || true
done
rm -rf "${PS:?}/Qt/qml" "${PS:?}/Qt/resources" "${PS:?}/Qt/translations" \
       "${PS:?}/Qt/libexec" "${PS:?}/Qt/metatypes" "${PS:?}/include" \
       "${PS:?}/typesystems" "${PS:?}/glue" "${PS:?}/scripts" \
       "${PS:?}/examples" "${PS:?}/doc" "${PS:?}/QtAsyncio"
rm -f "${PS:?}"/*.pyi "${PS:?}/py.typed"
# developer tools shipped next to the bindings (need glibc 2.34, unused here)
for t in assistant balsam balsamui designer linguist lrelease lupdate \
         qmlformat qmllint qmlls qsb svgtoqml; do
    rm -f "${PS:?}/$t"
done
# any remaining libraries of the removed module families (sub-libraries such
# as QuickControls2Imagine), the ffmpeg libs that only QtMultimedia uses
for g in Quick Designer Qml Web Multimedia 3D Sql Pdf Bluetooth Nfc Sensors \
         Positioning Location Scxml VirtualKeyboard Graphs Help ShaderTools; do
    rm -f "${PS:?}"/Qt/lib/libQt6"$g"*.so*
done
rm -f "${PS:?}"/Qt/lib/libav*.so* "${PS:?}"/Qt/lib/libsw*.so*
for d in qmltooling designer multimedia sensors position geoservices \
         scxmldatamodel texttospeech assetimporters geometryloaders \
         renderers renderplugins sceneparsers webview canbus qmllint \
         sqldrivers; do
    rm -rf "${PS:?}/Qt/plugins/$d"
done

# ---------------------------------------------------------------- payload
log "copying payload"
mkdir -p "$SHARE"
PAYLOAD=(mieworkbench scripts opticalproperties primitives basemodels docs share
         README.md INSTALL.md CUSTOMIZE.md CHANGELOG.md LICENSE miewb.env.example)
[ "${SKIP_DEMOS:-0}" = 1 ] || PAYLOAD+=(demos)
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$REPO" ls-files -z -- "${PAYLOAD[@]}"
else
    ( cd "$REPO" && find "${PAYLOAD[@]}" -type f -print0 )
fi | { grep -zEv '(^|/)(tests?|__pycache__)/|^docs/archive/|\.FCBak$|^demos/library_tests/generated/' || true; } \
  | tar -C "$REPO" --null -T - -cf - | tar -C "$SHARE" -xf -
# The repo-root developer operator notes ($NOTES_MD) are not shipped;
# rewrite cross-references to them in the copied payload (the sources
# themselves stay untouched), then refuse to build if any remain.
NOTES_MD="CLAUDE.md"
NOTES_STEM="${NOTES_MD%.md}"
grep -rIlZ -F "$NOTES_MD" "$SHARE" 2>/dev/null \
  | xargs -0 -r sed -i -e "s/${NOTES_STEM}\.md's/the developer notes'/g" \
                       -e "s/${NOTES_STEM}\.md/the developer notes/g"
if grep -rIil -F "$NOTES_STEM" "$SHARE" >/dev/null 2>&1; then
    echo "build_appimage: payload still references $NOTES_MD:" >&2
    grep -rIil -F "$NOTES_STEM" "$SHARE" >&2
    exit 1
fi
echo "$VERSION" >"$SHARE/VERSION"
# make the bundled sources importable without PYTHONPATH (relative to site dir)
echo "../../../share/mieworkbench" >"$SP/mieworkbench-app.pth"
# 'python3' shim for setup_env.sh (PATH is otherwise left alone)
mkdir -p "$SHARE/.shim"
ln -sf ../../../bin/python3 "$SHARE/.shim/python3"

# ------------------------------------------------------------------- trim
log "trimming caches and development files"
for p in conda-meta include share/man share/doc share/info lib/cmake \
         lib/pkgconfig ssl/man man compiler_compat; do
    rm -rf "${USR:?}/$p"
done
rm -rf "${USR:?}"/lib/tcl* "${USR:?}"/lib/tk* "${USR:?}"/lib/itcl* \
       "${USR:?}"/lib/thread* "${USR:?}"/lib/libtcl* "${USR:?}"/lib/libtk* \
       "${USR:?}"/lib/python3.*/test "${USR:?}"/lib/python3.*/idlelib \
       "${USR:?}"/lib/python3.*/tkinter "${USR:?}"/lib/python3.*/turtledemo \
       "${USR:?}"/lib/python3.*/lib2to3 "${USR:?}"/lib/python3.*/config-* \
       "${USR:?}"/lib/python3.*/lib-dynload/_tkinter* \
       "${SP:?}"/pip "${SP:?}"/pip-*.dist-info "${SP:?}"/setuptools \
       "${SP:?}"/setuptools-* "${SP:?}"/wheel "${SP:?}"/wheel-* \
       "${SP:?}"/pkg_resources
find "$USR" -name '*.a' -delete
find "$SP" -type d -name tests -prune -exec rm -rf {} +
find "$USR" -name __pycache__ -prune -exec rm -rf {} +
# precompile (unchecked-hash: valid regardless of mtimes on the read-only mount)
PYLIB="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["stdlib"])')"
"$PY" -W ignore -m compileall -q -j 0 --invalidation-mode unchecked-hash \
    "$PYLIB" "$SHARE/mieworkbench" "$SHARE/scripts" >/dev/null || true
rm -f "$USR"/bin/pip "$USR"/bin/pip3* "$USR"/bin/wheel "$USR"/bin/2to3* \
      "$USR"/bin/idle3* "$USR"/bin/pydoc3* "$USR"/bin/python3*-config

# -------------------------------------------------------------- AppDir files
log "writing AppRun, desktop file, icon"
install -m 755 "$HERE/AppRun" "$APPDIR/AppRun"
cat >"$APPDIR/mieworkbench.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=MieWorkbench
GenericName=Optical Ray Tracing Workbench
Comment=Design, simulate and analyze optical systems (FreeCAD-driven Mie ray tracer)
Exec=mieworkbench %f
Icon=mieworkbench
Terminal=false
Categories=Science;Physics;
MimeType=application/x-mieworkbench;application/x-miesim;application/x-extension-fcstd;
Keywords=optics;raytracing;lens;simulation;FreeCAD;
X-AppImage-Version=$VERSION
DESK
cp "$REPO/share/mieworkbench.svg" "$APPDIR/mieworkbench.svg"
ln -sf mieworkbench.svg "$APPDIR/.DirIcon"
mkdir -p "$USR/share/icons/hicolor/scalable/apps" "$USR/share/applications"
cp "$REPO/share/mieworkbench.svg" "$USR/share/icons/hicolor/scalable/apps/"
cp "$APPDIR/mieworkbench.desktop" "$USR/share/applications/"
if command -v desktop-file-validate >/dev/null; then
    desktop-file-validate "$APPDIR/mieworkbench.desktop" || die "invalid .desktop"
fi

# ------------------------------------------------------------------ checks
log "checking glibc requirements (max allowed $GLIBC_MAX)"
"$PY" -I - "$APPDIR" "$GLIBC_MAX" <<'PYEOF'
import os, re, subprocess, sys
root, limit = sys.argv[1], tuple(int(x) for x in sys.argv[2].split("."))
pat = re.compile(rb"GLIBC_(\d+)\.(\d+)")
worst, where, bad = (0, 0), "", []
for dp, dn, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        try:
            with open(p, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
        except OSError:
            continue
        try:
            out = subprocess.run(["objdump", "-T", p], capture_output=True,
                                 timeout=120).stdout
        except FileNotFoundError:
            print("objdump not found; skipping glibc scan")
            sys.exit(0)
        for m in pat.finditer(out):
            v = (int(m.group(1)), int(m.group(2)))
            if v > worst:
                worst, where = v, p
            if v > limit and p not in [b[1] for b in bad]:
                bad.append((v, p))
print("max GLIBC required: %d.%d (%s)" % (worst + (os.path.relpath(where, root),)))
for v, p in bad:
    print("  TOO NEW %d.%d  %s" % (v + (os.path.relpath(p, root),)))
sys.exit(1 if bad else 0)
PYEOF

log "smoke-testing the AppDir"
TMPHOME="$(mktemp -d "$WORK/home.XXXXXX")"
HOME="$TMPHOME" "$APPDIR/AppRun" --version
HOME="$TMPHOME" "$APPDIR/AppRun" python -I -c \
    "import numpy, scipy, h5py, PySide6; print('core python deps ok')"
# Qt/VTK need the host's OpenGL/EGL libraries; a build host without them
# only gets a warning (CI installs them and tests the GUI import).
HOME="$TMPHOME" QT_QPA_PLATFORM=offscreen "$APPDIR/AppRun" python -I -c \
    "import vtk; from PySide6 import QtCharts, QtWidgets, QtOpenGLWidgets; print('GUI python deps ok')" \
    || echo "WARNING: Qt/VTK import failed on this build host (missing system GL/EGL libraries?)" >&2
HOME="$TMPHOME" "$APPDIR/AppRun" trace --help >/dev/null 2>&1 || true
rm -rf "${TMPHOME:?}"

# ----------------------------------------------------------------- packing
OUT="$WORK/$OUT_NAME"
rm -f "${OUT:?}"
export APPIMAGE_EXTRACT_AND_RUN=1
if [ -n "${APPIMAGETOOL:-}" ] || [ -z "${RUNTIME_FILE:-}" ]; then
    TOOL="${APPIMAGETOOL:-}"
    if [ -z "$TOOL" ]; then
        TOOL="$TOOLS/appimagetool-$ARCH.AppImage"
        if [ ! -x "$TOOL" ]; then
            log "downloading appimagetool"
            curl -fsSL -o "$TOOL" \
                "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage" \
                || die "could not download appimagetool; set APPIMAGETOOL=<path> or RUNTIME_FILE=<type2 runtime>"
            chmod +x "$TOOL"
        fi
    fi
    log "packing with appimagetool"
    args=()
    [ -z "${RUNTIME_FILE:-}" ] || args+=(--runtime-file "$RUNTIME_FILE")
    ARCH=$ARCH "$TOOL" "${args[@]}" "$APPDIR" "$OUT"
else
    log "packing with mksquashfs + runtime $RUNTIME_FILE"
    [ -f "$RUNTIME_FILE" ] || die "RUNTIME_FILE not found: $RUNTIME_FILE"
    MKSQ="$(command -v mksquashfs || true)"
    if [ -z "$MKSQ" ]; then
        mm create -y -q -p "$WORK/sqenv" -c conda-forge --override-channels \
            squashfs-tools >/dev/null
        MKSQ="$WORK/sqenv/bin/mksquashfs"
    fi
    "$MKSQ" "$APPDIR" "$WORK/image.squashfs" -root-owned -noappend \
        -comp "${SQUASH_COMP:-zstd}" -b 256K -no-progress >/dev/null
    cat "$RUNTIME_FILE" "$WORK/image.squashfs" >"$OUT"
    chmod 755 "$OUT"
    rm -f "${WORK:?}/image.squashfs"
fi
[ -s "$OUT" ] || die "packing produced no output"

# ----------------------------------------------------------------- cleanup
if [ "${KEEP_CACHE:-0}" != 1 ]; then
    log "removing intermediates"
    rm -rf "${MAMBA_ROOT:?}" "${BUILDENV:?}" "${PIPCACHE:?}" "${WORK:?}/cengine-build" "${WORK:?}/sqenv"
fi
log "done: $OUT ($(du -h "$OUT" | cut -f1))"
