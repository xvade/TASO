#!/bin/bash
# Build the min/max/sub-extended generator (run inside tensat.sif).
# Regenerate protobuf against the container's protoc first (checked-in rules.pb.*
# are from an older protoc), then compile against the minimal xflow/ops.h shim.
# -DPWL_FOCUS restricts the op set to the piecewise-linear family (fast; depth-4 priceable).
set -e
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH; export PATH=/opt/conda/bin:$PATH
protoc -I ../core --cpp_out=. ../core/rules.proto
g++ generator.cc rules.pb.cc -o generator_pwl \
  -I ../../include -I/opt/conda/include -L/opt/conda/lib \
  -lprotobuf -std=c++11 -pthread -O2 -DPWL_FOCUS
