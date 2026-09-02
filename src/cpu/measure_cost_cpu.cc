/* Synthetic (non-hardware) op cost model.
 *
 * This file replaces the 22 Model::measure_*_cost functions that
 * src/cudnn (the per-op _kernel.cu files) implement by actually running
 * each op on a GPU and timing it with cudnn/cuda events. When TASO is
 * built with USE_CUDA=OFF there is no such backend, so these
 * implementations instead estimate a cost analytically from tensor
 * shapes: (flops + mem_acc) / UNITS_PER_MS, mirroring the "flops" and
 * "mem_acc" formulas each op class already computes in its own
 * (CUDA-independent) collect_costs(), over in src/core.
 *
 * INVARIANT (2026-09-02): this project NEVER uses TASO's op runtime cost.
 * It is not about runtime efficiency -- extraction is verifiability-driven
 * (tensat's VerifCost / IBP interval-gap area), not runtime-driven. So op
 * cost is force-zeroed here: taso performs no cost measurement, and every
 * op->runtime is 0. This is the ONE reason a cost backend mattered to us at
 * all (op creation calls measure_*_cost); zeroing it also means the cudnn
 * build's cublas SGEMM at op-creation-time -- the small-N abort -- never
 * fires via this path. Define TASO_ENABLE_COST_MEASUREMENT to restore the
 * synthetic analytic estimate for a one-off runtime-cost experiment.
 */

#include "taso/ops.h"
using namespace taso;

namespace {

const float UNITS_PER_MS = 1e9f;

inline float estimate_runtime(float flops, float mem_acc)
{
  // Zeroed by the file-level invariant above. Safe: tensat reads op->runtime
  // in CostModel::get_self_cost but never divides by it or asserts it > 0, and
  // our pipeline extracts with VerifCost, not the runtime-driven CostModel.
#ifdef TASO_ENABLE_COST_MEASUREMENT
  return (flops + mem_acc) / UNITS_PER_MS;
#else
  (void) flops;
  (void) mem_acc;
  return 0.0f;
#endif
}

inline int volume(const Tensor& t)
{
  int v = 1;
  for (int i = 0; i < t.numDim; i++)
    v *= t.dim[i];
  return v;
}

} // namespace

void Model::measure_conv2d_cost(Conv2D* conv)
{
  int outputSize = volume(conv->outputs[0]);
  int kernelH = conv->inputs[1].dim[2];
  int kernelW = conv->inputs[1].dim[3];
  int inputC = conv->inputs[1].dim[1];
  float flops = (float)outputSize * (kernelH * kernelW * inputC + 1);
  if (conv->activation != AC_MODE_NONE)
    flops += outputSize;
  float mem_acc = volume(conv->inputs[0]) + outputSize + volume(conv->inputs[1]);
  conv->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Conv2D]: cost(%.4lf)\n", conv->runtime);
}

void Model::measure_matmul_cost(Matmul* mm)
{
  int outputSize = volume(mm->outputs[0]);
  int inputSize = volume(mm->inputs[0]);
  float flops = (float)outputSize * mm->inputs[0].dim[mm->inputs[0].numDim - 1];
  float mem_acc = inputSize;
  mm->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Matmul]: cost(%.4lf)\n", mm->runtime);
}

void Model::measure_mul_cost(Mul* mul)
{
  // Mul::collect_costs() is itself unimplemented upstream (TODO), so
  // there's no existing formula to mirror; treat it like a generic
  // elementwise op instead.
  float flops = volume(mul->outputs[0]);
  float mem_acc = volume(mul->inputs[0]);
  mul->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Mul]: cost(%.4lf)\n", mul->runtime);
}

void Model::measure_pad_cost(Pad* pad)
{
  float flops = volume(pad->inputs[0]);
  float mem_acc = volume(pad->inputs[0]) + volume(pad->outputs[0]);
  pad->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Pad]: cost(%.4lf)\n", pad->runtime);
}

void Model::measure_pool2d_cost(Pool2D* pool)
{
  int outputSize = volume(pool->outputs[0]);
  float flops = (float)outputSize * pool->kernelH * pool->kernelW;
  float mem_acc = volume(pool->inputs[0]);
  pool->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Pool2D]: cost(%.4lf)\n", pool->runtime);
}

void Model::measure_topk_cost(TopK* topk)
{
  // TASO's own collect_costs() treats TopK as free (no flops/mem_acc).
  topk->runtime = 0;
  if (print_cost)
    printf("        measure[TopK]: cost(%.4lf)\n", topk->runtime);
}

void Model::measure_transpose_cost(Transpose* transpose)
{
  if (transpose->shuffle) {
    float v = volume(transpose->outputs[0]);
    transpose->runtime = estimate_runtime(v, v);
  } else {
    transpose->runtime = 0;
  }
  if (print_cost)
    printf("        measure[Transpose]: cost(%.4lf)\n", transpose->runtime);
}

void Model::measure_reduce_cost(Reduce* reduce)
{
  float flops = volume(reduce->inputs[0]);
  float mem_acc = volume(reduce->inputs[0]) + volume(reduce->outputs[0]);
  reduce->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Reduce]: cost(%.4lf)\n", reduce->runtime);
}

void Model::measure_reshape_cost(Reshape* reshape)
{
  // Pure metadata op in TASO's own cost accounting: no flops/mem_acc.
  reshape->runtime = 0;
  if (print_cost)
    printf("        measure[Reshape]: cost(%.4lf)\n", reshape->runtime);
}

void Model::measure_resize_cost(Resize* resize)
{
  resize->runtime = 0;
  if (print_cost)
    printf("        measure[Resize]: cost(%.4lf)\n", resize->runtime);
}

void Model::measure_activation_cost(Activation* activation)
{
  int outputSize = volume(activation->outputs[0]);
  float flops = (activation->type == OP_RELU) ? 0.0f : (float)outputSize;
  float mem_acc = volume(activation->inputs[0]);
  activation->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Activation]: cost(%.4lf)\n", activation->runtime);
}

void Model::measure_batchnorm_cost(BatchNorm* bn)
{
  int outputSize = volume(bn->outputs[0]);
  float flops = (float)outputSize * 2;
  float mem_acc = volume(bn->inputs[0]);
  bn->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[BatchNorm]: cost(%.4lf)\n", bn->runtime);
}

void Model::measure_cast_cost(Cast* cast)
{
  float flops = volume(cast->outputs[0]);
  float mem_acc = volume(cast->inputs[0]);
  cast->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Cast]: cost(%.4lf)\n", cast->runtime);
}

void Model::measure_concat_cost(Concat* concat)
{
  float mem_acc = 0;
  for (int i = 0; i < concat->numInputs; i++)
    if (concat->needCopy[i])
      mem_acc += volume(concat->inputs[i]);
  concat->runtime = estimate_runtime(0.0f, mem_acc);
  if (print_cost)
    printf("        measure[Concat]: cost(%.4lf)\n", concat->runtime);
}

void Model::measure_shape_cost(Shape* shape)
{
  shape->runtime = 0;
  if (print_cost)
    printf("        measure[Shape]: cost(%.4lf)\n", shape->runtime);
}

void Model::measure_slice_cost(Slice* slice)
{
  slice->runtime = 0;
  if (print_cost)
    printf("        measure[Slice]: cost(%.4lf)\n", slice->runtime);
}

void Model::measure_element_cost(Element* element)
{
  int outputSize = volume(element->outputs[0]);
  int inputSize = volume(element->inputs[0]);
  float flops = outputSize;
  float mem_acc = inputSize * 2;
  element->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Element]: cost(%.4lf)\n", element->runtime);
}

void Model::measure_elementwise_unary_cost(ElementWiseUnary* elu)
{
  float flops = volume(elu->outputs[0]);
  float mem_acc = volume(elu->inputs[0]);
  elu->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[ElementWiseUnary]: cost(%.4lf)\n", elu->runtime);
}

void Model::measure_enlarge_cost(Enlarge* enlarge)
{
  int outputSize = volume(enlarge->outputs[0]);
  int inputSize = volume(enlarge->inputs[0]);
  float flops = outputSize;
  float mem_acc = inputSize + outputSize;
  enlarge->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Enlarge]: cost(%.4lf)\n", enlarge->runtime);
}

void Model::measure_squeeze_cost(Squeeze* squeeze)
{
  squeeze->runtime = 0;
  if (print_cost)
    printf("        measure[Squeeze]: cost(%.4lf)\n", squeeze->runtime);
}

void Model::measure_unsqueeze_cost(Unsqueeze* unsqueeze)
{
  unsqueeze->runtime = 0;
  if (print_cost)
    printf("        measure[Unsqueeze]: cost(%.4lf)\n", unsqueeze->runtime);
}

void Model::measure_where_cost(Where* where)
{
  float flops = volume(where->outputs[0]);
  float mem_acc = 4.0f * volume(where->outputs[0]);
  where->runtime = estimate_runtime(flops, mem_acc);
  if (print_cost)
    printf("        measure[Where]: cost(%.4lf)\n", where->runtime);
}
