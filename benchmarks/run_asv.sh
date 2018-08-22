pip install asv

if [[ ! -z "$CI" ]]; then
  mkdir -p ~/.sumpy/asv
  ln -s ~/.sumpy/asv .asv
fi

asv machine --yes
asv setup --verbose
master_commit=`git rev-parse master`
test_commit=`git rev-parse HEAD`

export PYOPENCL_CTX=port

asv run $master_commit...$master_commit~ --skip-existing --verbose
asv run $test_commit...$test_commit~ --skip-existing --verbose

output=`asv compare $master_commit $test_commit --factor 1 -s`
echo "$output"

if [[ "$output" = *"worse"* ]]; then
  echo "Some of the benchmarks have gotten worse"
  exit 1
fi

if [[ ! -z "$CI" ]]; then
  asv publish --html-dir ~/.scicomp-benchmarks/asv/sumpy
fi
