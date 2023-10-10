# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux2014_x86_64

RUN yum install git -y
RUN git config --global --add safe.directory /code

# Download rustup-init asn install
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.65

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUST_BUILD_TARGET="x86_64-unknown-linux-gnu"
