/* Execution-method stubs for the CPU-only build.
 *
 * Every concrete Op subclass declares map()/unmap()/forward() as overrides
 * of OpBase's pure-virtual methods (include/taso/ops.h). Those overrides
 * are only defined in the per-op _kernel.cu files under src/cudnn (real
 * cudnn/cuda execution) for every op except NoOp and Split, which already
 * have CPU-native implementations in src/core/{noop,split}.cc.
 *
 * Even though TENSAT's search/extraction pipeline never calls these
 * methods -- it only builds graph structure and reads per-op cost via
 * measure_*_cost (see measure_cost_cpu.cc) -- C++ still needs a linkable
 * function address for every vtable slot of every class that gets
 * instantiated (e.g. via Model::get_or_create_conv2d), regardless of
 * whether that slot is ever actually invoked at runtime. Without these
 * stubs, linking libtaso_runtime with USE_CUDA=OFF fails with "undefined
 * reference" for each op's map/unmap/forward.
 *
 * These stubs intentionally fail loudly (not silently no-op) if ever
 * actually called, since that would mean some code path we didn't
 * anticipate is trying to execute a real op, which this CPU-only build
 * cannot do correctly.
 */

#include "taso/ops.h"
#include <cstdio>
#include <cstdlib>
using namespace taso;

namespace {
void unimplemented(const char* op_name, const char* method)
{
  fprintf(stderr,
      "taso (CPU-only build, USE_CUDA=OFF): %s::%s() was called, but this "
      "build has no real op execution -- only cost estimation and graph "
      "structure are supported. Rebuild with USE_CUDA=ON if real execution "
      "is needed.\n", op_name, method);
  abort();
}
} // namespace

#define STUB_EXEC(CLASS) \
  void CLASS::map(void) {} \
  void CLASS::unmap(void) {} \
  void CLASS::forward(bool block) { unimplemented(#CLASS, "forward"); }

STUB_EXEC(Constant)
STUB_EXEC(Conv2D)
STUB_EXEC(Matmul)
STUB_EXEC(Mul)
STUB_EXEC(Pool2D)
STUB_EXEC(Activation)
STUB_EXEC(BatchNorm)
STUB_EXEC(Cast)
STUB_EXEC(Concat)
STUB_EXEC(Element)
STUB_EXEC(ElementWiseUnary)
STUB_EXEC(Enlarge)
STUB_EXEC(TopK)
STUB_EXEC(MergeGConv)
STUB_EXEC(Pad)
STUB_EXEC(Reduce)
STUB_EXEC(Reshape)
STUB_EXEC(Resize)
STUB_EXEC(Shape)
STUB_EXEC(Slice)
STUB_EXEC(Squeeze)
STUB_EXEC(Transpose)
STUB_EXEC(Unsqueeze)
STUB_EXEC(Where)

#undef STUB_EXEC
