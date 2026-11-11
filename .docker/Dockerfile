FROM ubuntu:24.04

RUN apt update && \
    apt install -y --no-install-recommends \
    ca-certificates \
    sudo \
    tini \
    git \
    vim && \
    rm -rf /var/lib/apt/lists/*

ARG NB_USER="tau"
ARG NB_UID="1000"
ARG NB_GID="100"

ENV HOME="/home/${NB_USER}"

# Delete existing user with UID="${NB_UID}" if it exists
RUN if grep -q "${NB_UID}" /etc/passwd; then \
        userdel --remove $(id -un "${NB_UID}"); \
    fi

# Create a non-root user and allow sudo access
RUN useradd --no-log-init --create-home --shell /bin/bash --no-user-group --gid $NB_GID --uid $NB_UID  $NB_USER \
&& chown -R $NB_UID:$NB_GID $HOME
ARG WITH_SUDO="no"
RUN if [ "$WITH_SUDO" = "yes" ]; then \
  echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
;fi

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 \
/uv /uvx /usr/local/bin/
ENV PATH="/usr/local/bin:${PATH}"

# Install Python 3.13 and packages
USER $NB_USER
WORKDIR /workspace
RUN uv python install 3.13
RUN git clone --depth 1 --single-branch --branch main https://github.com/sierra-research/tau2-bench && \
    cd tau2-bench && \
    uv sync 
ENV PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace/tau2-bench
ENTRYPOINT ["/bin/tini", "-s", "--"]
CMD ["/bin/bash"]