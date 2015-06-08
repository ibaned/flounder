#!/bin/bash -ex
perf script | ../FlameGraph/stackcollapse-perf.pl > out.perf-folded
../FlameGraph/flamegraph.pl --title=flounder out.perf-folded > perf-test.svg
