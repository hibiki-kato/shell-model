FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh \
    && apt update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y \
    libeigen3-dev \
    libpython3-all-dev \
    python3-numpy \
    python3-matplotlib
    # libboost-all-dev \
    # build-essential \
    # gnuplot \
    # libpython2.7-dev \

    # language-pack-ja-base \
    # language-pack-ja \
    # locales \
    # && locale-gen ja-JP.UTF-8

# #言語を日本語に設定
# ENV LANG ja_JP.UTF-8∏


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
