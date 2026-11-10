# llama-server b11510 (llama.cpp c35b66744f13cb0dcc476af063e112122eee9355), built as on the host:
# CUDA 13.0.3, GCC 13, CMake 3.28, -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
# -DCMAKE_BUILD_TYPE=Release and every other option at its default (GGML_NATIVE=ON, so this image
# is tuned to the CPU of the machine that builds it). The build context is this folder only.
#
# The build stage records the toolchain versions in /opt/llama/build-info/ and keeps the
# CMakeCache.txt there, so scripts can diff it against the host's ~/llama.cpp/build.

# --- build -------------------------------------------------------------------------------------
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS build

ARG LLAMA_COMMIT=c35b66744f13cb0dcc476af063e112122eee9355
ARG BUILD_JOBS=8

# The host build's tools: cmake 3.28.3, gcc/g++ 13.3.0, git, make, pkg-config, libssl-dev
# (LLAMA_OPENSSL=ON), libgomp1 (GGML_OPENMP=ON). No ccache (the host has none).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake gcc-13 g++-13 git libssl-dev libgomp1 make pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/llama/build-info \
 && dpkg-query -W -f='${Package} ${Version}\n' \
      build-essential cmake gcc-13 g++-13 git libssl-dev libgomp1 make pkg-config \
      > /opt/llama/build-info/packages.txt \
 && { c++ --version | head -1; cmake --version | head -1; nvcc --version | tail -2; } \
      > /opt/llama/build-info/toolchain.txt \
 && cat /opt/llama/build-info/packages.txt /opt/llama/build-info/toolchain.txt \
 && grep -q ' 13.3.0' /opt/llama/build-info/toolchain.txt \
 && grep -q 'cmake version 3.28.3' /opt/llama/build-info/toolchain.txt \
 && grep -q 'V13.0.88' /opt/llama/build-info/toolchain.txt

# Full commit history (blobless) so the build number (git rev-list --count) is 11510.
WORKDIR /src
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git llama.cpp \
 && cd llama.cpp \
 && git checkout --detach "$LLAMA_COMMIT" \
 && test "$(git rev-parse HEAD)" = "$LLAMA_COMMIT" \
 && test "$(git rev-list --count HEAD)" = 11510 \
 && git describe --tags --always | tee /opt/llama/build-info/git-describe.txt

# Link time only: the driver library is not in a build container, so the CUDA stub is offered
# as libcuda.so.1 through LD_LIBRARY_PATH for this step (the stage is discarded). This avoids
# any non-default linker flag; the runtime image gets the real driver from the nvidia runtime.
WORKDIR /src/llama.cpp
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
 && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target llama-server -j "$BUILD_JOBS" \
 && cp build/CMakeCache.txt /opt/llama/build-info/CMakeCache.txt \
 && mkdir -p /opt/llama/bin \
 && cp -P build/bin/llama-server build/bin/*.so* /opt/llama/bin/

# --- runtime -----------------------------------------------------------------------------------
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:76f46f3e649824ebd3685d3d86c42b0ea62af04f59242f419f56cd5804a72efb

ARG APP_HOME=/home/cal

# libgomp1 (OpenMP) and libssl3t64 (libssl.so.3, libcrypto.so.3): what the host binaries link
# besides CUDA and libc. The base image's ubuntu user (uid/gid 1000) becomes "app" with
# HOME=$APP_HOME, so the bind mounts' absolute paths and the user match the host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libssl3t64 \
 && rm -rf /var/lib/apt/lists/* \
 && groupmod -n app ubuntu \
 && usermod -l app -d "$APP_HOME" -m ubuntu \
 && test "$(id -u app):$(id -g app)" = 1000:1000

COPY --from=build /opt/llama/bin/ /opt/llama/bin/
COPY --from=build /opt/llama/build-info/ /opt/llama/build-info/

ENV PATH=/opt/llama/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama/bin \
    HOME=$APP_HOME

LABEL org.opencontainers.image.title="hs-rag-classifier llama-server" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="c35b66744f13cb0dcc476af063e112122eee9355" \
      org.opencontainers.image.version="b11510"

USER 1000:1000
WORKDIR $APP_HOME
ENTRYPOINT ["llama-server"]
