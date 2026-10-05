#!/bin/sh
# Run inside Alpine on the target architecture. Never install a setuid helper.
set -eu
: "${BWRAP_SHA256:?Supply the independently reviewed upstream source SHA-256}"
version=0.11.0
arch=$(uname -m)
case "$arch" in x86_64|aarch64) ;; *) echo "Unsupported architecture: $arch" >&2; exit 1;; esac
apk add --no-cache build-base meson ninja pkgconf curl libcap-dev libcap-static python3 py3-build py3-setuptools py3-wheel
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
url="https://github.com/containers/bubblewrap/releases/download/v$version/bubblewrap-$version.tar.xz"
curl --fail --location --proto '=https' "$url" -o "$work/source.tar.xz"
printf '%s  %s\n' "$BWRAP_SHA256" "$work/source.tar.xz" | sha256sum -c -
tar -xf "$work/source.tar.xz" -C "$work"
meson setup "$work/build" "$work/bubblewrap-$version" -Dman=disabled -Dprefer_static=true -Dc_link_args=-static
meson compile -C "$work/build"
out="remie/_vendor/bubblewrap/linux-$arch"
mkdir -p "$out"
install -m 0755 "$work/build/bwrap" "$out/bwrap"
cp "$work/bubblewrap-$version/COPYING" remie/_vendor/bubblewrap/COPYING
printf 'Bubblewrap %s\nSource: %s\nSHA256: %s\n' "$version" "$url" "$BWRAP_SHA256" > remie/_vendor/bubblewrap/SOURCE
# A static executable must not request an ELF dynamic interpreter.
if readelf -l "$out/bwrap" | grep -q INTERP; then
    echo 'Expected a static Bubblewrap executable' >&2; exit 1
fi
"$out/bwrap" --version
python3 -m build --wheel --no-isolation
