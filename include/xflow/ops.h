// Minimal enum/struct-only XFlow header for building src/generator/generator.cc
// standalone (the original pulled in cudnn/TensorRT/MKL, none of which the
// generator needs -- it defines its own OpTemp/TensorTemp and only uses these
// enums + SplitInfo). Enums copied verbatim from the original xflow/ops.h
// (recovered from git 34d0138) so op-type ints stay compatible with the existing
// graph_subst.pb / tensat pipeline. OP_EW_SUB/MAX/MIN appended (ints 26/27/28).
#ifndef _XFLOW_OPS_MINIMAL_H_
#define _XFLOW_OPS_MINIMAL_H_
#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
using namespace std;  // generator.cc uses bare max()/min() (as the original did)

#define MAX_DIM 4
#define MAX_NUM_SPLITS 32
#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6

namespace XFlow{

enum OpType {
  OP_INPUT,
  OP_WEIGHT,
  OP_ANY,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_POOL2D_MAX,
  OP_POOL2D_AVG,
  OP_RELU,
  OP_SIGMOID,
  OP_TANH,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_RESHAPE,
  OP_TRANSPOSE,
  OP_EW_ADD,
  OP_EW_MUL,
  OP_MATMUL,
  OP_MUL,
  OP_ENLARGE,
  OP_MERGE_GCONV,
  OP_CONSTANT_IMM,
  OP_CONSTANT_ICONV,
  OP_CONSTANT_ONE,
  OP_CONSTANT_POOL,
  OP_EW_SUB, // appended for verifiability (PWL) rewrites; ONNX Sub
  OP_EW_MAX, // ONNX Max
  OP_EW_MIN, // ONNX Min
};

enum ActiMode {
  AC_MODE_NONE,
  AC_MODE_SIGMOID,
  AC_MODE_RELU,
  AC_MODE_TANH,
};

enum PaddingMode {
  PD_MODE_SAME,
  PD_MODE_VALID,
};

enum PMParameter {
  PM_OP_TYPE,
  PM_NUM_INPUTS,
  PM_NUM_OUTPUTS,
  PM_GROUP,
  PM_KERNEL_H,
  PM_KERNEL_W,
  PM_STRIDE_H,
  PM_STRIDE_W,
  PM_PAD,
  PM_ACTI,
  PM_NUMDIM,
  PM_AXIS,
  PM_PERM,
  PM_OUTSHUFFLE,
  PM_MERGE_GCONV_COUNT,
};

struct SplitInfo {
  SplitInfo(void) {num = 0;}
  inline bool operator==(const SplitInfo& rhs) const {
    if (num != rhs.num) return false;
    for (int i = 0; i < num; i++)
      if (pos[i] != rhs.pos[i])
        return false;
    return true;
  }
  void merge(int offset, const SplitInfo& next) {
    if (num + 1 + next.num >= MAX_NUM_SPLITS) {
      printf("num = %d, next.num = %d\n", num, next.num);
    }
    assert(num + 1 + next.num < MAX_NUM_SPLITS);
    for (int i = 0; i < next.num; i++)
      pos[num++] = offset + next.pos[i];
    pos[num++] = offset;
  }
  inline bool operator!=(const SplitInfo& rhs) const
  {
    if (num != rhs.num) return true;
    for (int i = 0; i < num; i++)
      if (pos[i] != rhs.pos[i]) return true;
    return false;
  }
  SplitInfo& operator=(const SplitInfo& st)
  {
    num = st.num;
    for (int i = 0; i < num; i++)
      pos[i] = st.pos[i];
    return *this;
  }
  void divide(SplitInfo& left, SplitInfo& right, int &mid) {
    assert(num > 0);
    left.num = 0;
    right.num = 0;
    mid = pos[num - 1];
    int idx = 0;
    while (idx < num && pos[idx] < mid)
      left.pos[left.num++] = pos[idx++];
    while (idx < num - 1)
      right.pos[right.num++] = pos[idx++];
  }
  void combine(const SplitInfo& next) {
    if (num != next.num)
      num = 0;
    for (int i = 0; i < num; i++)
      if (pos[i] != next.pos[i]) {
        num = 0;
        return;
      }
  }
  void serialize(int* keys, int& idx) const {
    keys[idx++] = num;
    for (int i = 0; i < num; i++)
      keys[idx++] = pos[i];
  }
  static const SplitInfo NO_SPLIT;
  int num;
  int pos[MAX_NUM_SPLITS];
};

} // namespace XFlow
#endif
