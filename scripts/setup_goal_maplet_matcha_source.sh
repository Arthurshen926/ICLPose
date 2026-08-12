#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/tmp/matcha-gaussians-official}
project=/root/ICLPose
commit=b119fd96e484fc81eb40623c1ea92ad3dbd3c21e
patch=$project/configs/vfm/matcha_b119fd96_rtx3090_cuda116.patch

if [[ ! -d $repo/.git ]]; then
  echo "MAtCha checkout is absent: $repo" >&2
  exit 2
fi
actual=$(git -C "$repo" rev-parse HEAD)
if [[ $actual != "$commit" ]]; then
  echo "MAtCha commit drift: expected $commit, got $actual" >&2
  exit 2
fi
if [[ -n $(git -C "$repo" status --short --untracked-files=no) ]]; then
  if git -C "$repo" diff --binary --quiet --no-ext-diff; then
    echo "MAtCha index contains an unsupported staged change" >&2
    exit 2
  fi
  current=$(mktemp)
  trap 'rm -f "$current"' EXIT
  git -C "$repo" diff --binary --no-ext-diff > "$current"
  if cmp -s "$current" "$patch"; then
    echo "MAtCha source patch already matches the versioned contract"
    exit 0
  fi
  echo "MAtCha has tracked changes other than the versioned patch" >&2
  exit 2
fi
git -C "$repo" apply --check "$patch"
git -C "$repo" apply "$patch"
echo "Applied the versioned RTX3090/CUDA11.6 cuRoPE patch"
