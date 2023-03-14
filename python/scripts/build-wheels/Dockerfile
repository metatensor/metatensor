# Use manylinux docker image as a base
FROM quay.io/pypa/manylinux2010_x86_64

# Use the repositories from archive.kernel.org, the ones from vault.centos.org
# are returning 403 errors
RUN rm -f /etc/yum.repos.d/CentOS-{Debuginfo,fasttrack,Media,Vault,SCLo-scl,SCLo-scl-rh}.repo && \
    sed -i "s|vault.centos.org/centos/\$releasever|archive.kernel.org/centos-vault/6.9|g" /etc/yum.repos.d/CentOS-Base.repo

RUN yum install git -y
RUN git config --global --add safe.directory /code

# Manually download rustup-init version 1.25.1, since 1.25.2 requires a more
# modern glibc. We can remove this if https://github.com/rust-lang/rustup/issues/3203
# is solved by making rustup-init.sh able to detect the glibc version.
RUN curl -O https://static.rust-lang.org/rustup/archive/1.25.1/x86_64-unknown-linux-gnu/rustup-init && \
    chmod +x ./rustup-init && \
    ./rustup-init -y --profile minimal --default-toolchain 1.63

ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUST_BUILD_TARGET="x86_64-unknown-linux-gnu"
