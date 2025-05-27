# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux_2_28_x86_64

RUN yum install git -y
RUN git config --global --add safe.directory /code

# Download rustup-init asn install
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.86

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUST_BUILD_TARGET="x86_64-unknown-linux-gnu"

# Install an older C++ compiler. The default compiler (gcc-14) introduces calls
# to `__cxa_call_terminate` which is not available in ubuntu-22.04 libstdc++
ARG DEVTOOLSET_VERSION=11
RUN yum install -y gcc-toolset-${DEVTOOLSET_VERSION}-toolchain
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH
