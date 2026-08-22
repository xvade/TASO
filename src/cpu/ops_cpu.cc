/* CPU-only Model implementation.
 *
 * This file replaces src/cudnn/ops_cudnn.cu when TASO is built with
 * USE_CUDA=OFF. It provides host-memory-only versions of the handful of
 * Model methods that the CUDA build implements via cudaMalloc/cudnnCreate:
 * the constructor, allocate_memory/copy_memory, and measure_oplist_runtime.
 *
 * Nothing here does real tensor computation -- see measure_cost_cpu.cc and
 * execution_stubs_cpu.cc for why that's fine for TENSAT's optimizer/
 * extraction pipeline, which never executes ops, only measures their
 * (synthetic) cost and reasons about graph structure.
 */

#include "taso/ops.h"
#include <cstdlib>
#include <cstring>
using namespace taso;

Model::Model()
: isTraining(false), print_cost(false)
{
  global_unique_id = 100;
  // No cudnn/cublas handles, no device workspace or scratch buffers to
  // allocate: nothing in src/core/*.cc reads Model's GPU scratch fields
  // (inputPtr, outputPtr, dnn, blas, ...) outside of src/cudnn/*.cu, which
  // isn't compiled in this configuration.
  workSpaceSize = 0;
  workSpace = nullptr;
  inputPtr = nullptr;
  biasPtr = nullptr;
  outputPtr = nullptr;
  filterPtr = nullptr;
  scalePtr = nullptr;
  runningMean = nullptr;
  runningVar = nullptr;
  saveMean = nullptr;
  saveVar = nullptr;
}

void* Model::allocate_memory(size_t size, const DATATYPE* data_initial)
{
  void* ptr = malloc(size);
  if (data_initial != NULL) {
    memcpy(ptr, data_initial, size);
  }
  return ptr;
}

bool Model::copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size)
{
  memcpy(dst, src, size);
  return true;
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList)
{
  // The CUDA version of this function actually executes opBaseList
  // (forward()) repeatedly and times it with cuda events; forward() isn't
  // implemented in this CPU build (see execution_stubs_cpu.cc), so instead
  // we just sum each op's already-measured (synthetic) per-op runtime.
  // This is only used by Graph::run(), which TENSAT's search/extraction
  // path does not call.
  float total = 0.0f;
  for (size_t i = 0; i < opBaseList.size(); i++)
    total += opBaseList[i]->runtime;
  return total;
}
