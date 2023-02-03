# Use a many linux with a specification
# manylinux_${GLIBCMAJOR}_${GLIBCMINOR}_${ARCH}
FROM quay.io/pypa/manylinux_2_28_x86_64

RUN yum install git -y
RUN git config --global --add safe.directory /code

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUST_BUILD_TARGET="x86_64-unknown-linux-gnu"
