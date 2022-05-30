FROM quay.io/pypa/manylinux2010_x86_64

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUST_BUILD_TARGET="x86_64-unknown-linux-gnu"
