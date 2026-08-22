# Deferred: this project doesn't have a CUDA/GPU environment set up yet.
# Set both back to ON (and make sure a CUDA toolkit + cuDNN are installed)
# to switch back to real GPU execution and hardware-measured op costs --
# see src/cpu/*.cc for what USE_CUDA=OFF builds against instead.
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(CMAKE_BUILD_TYPE Debug)
