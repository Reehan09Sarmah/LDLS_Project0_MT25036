#!/bin/bash
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
L="$R/llama.cpp"
COMMIT=10a60cf303566e10d6a7a2774c17d2085503d87b
if [ ! -f "$L/CMakeLists.txt" ]; then
  mkdir -p "$L"
  git -C "$L" init -q
  git -C "$L" remote add origin https://github.com/ggml-org/llama.cpp 2>/dev/null || true
  git -C "$L" fetch -q --depth 1 origin "$COMMIT"
  git -C "$L" checkout -q FETCH_HEAD
fi
mkdir -p "$L/examples/tree-driver"
cp "$R/driver/tree_driver.cpp" "$R/driver/CMakeLists.txt" "$L/examples/tree-driver/"
python3 - "$L/examples/CMakeLists.txt" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()
if "add_subdirectory(tree-driver)" not in text:
    text = re.sub(r"^(\s*)add_subdirectory\(batched\)\n", r"\g<0>\1add_subdirectory(tree-driver)\n", text, count=1, flags=re.M)
    open(path, "w").write(text)
PY
grep -q "add_subdirectory(tree-driver)" "$L/examples/CMakeLists.txt" || { echo "could not register tree-driver"; exit 1; }
cmake -S "$L" -B "$L/build" -DGGML_CUDA=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$L/build" --config Release -j "${JOBS:-8}" --target tree-driver llama-tokenize llama-batched llama-simple
ls "$L/build/bin/" | grep -E "^(tree-driver|llama-tokenize|llama-batched|llama-simple)$"
