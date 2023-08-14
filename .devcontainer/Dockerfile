FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.27.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh \
    && apt update \
    && apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y \
    build-essential \
    libeigen3-dev \
    libpython3-all-dev \
    python3-numpy \
    python3-matplotlib \
    msttcorefonts 

    # refresh matplotlib cache
    # && rm -r ~/.cache/matplotlib


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
