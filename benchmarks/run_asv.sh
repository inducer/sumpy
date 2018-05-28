pip install asv
asv setup --verbose
master_commit=`git rev-parse master`
test_commit=`git rev-parse HEAD`

export PYOPENCL_CTX=0

asv run $master_commit...$master_commit~ --skip-existing --verbose
asv run $test_commit...$test_commit~ --skip-existing --verbose

output=`asv compare $master_commit $test_commit --factor 1 -s`
echo "$output"

if [[ "$output" = *"worse"* ]]; then
  echo "Some of the benchmarks have gotten worse"
fi
