# Dev Container: DS Ubuntu 22. Includes miniconda, git and poetry
# Angel Martinez Tenor 2022

FROM ubuntu:22.04

# Root user (replicate in commands line with sudo) ---------------------------------------------------------------------

# From https://hub.docker.com/r/continuumio/miniconda/dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing -y && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends git make curl wget bzip2 ca-certificates libglib2.0-0 libxext6 \
  libsm6 libxrender1 mercurial subversion nano htop unzip

# # INSTALL MINICONDA (root)
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#   /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#   rm ~/miniconda.sh && \
#   ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#   echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#   echo "conda activate base" >> ~/.bashrc

# CREATE AND ACTIVATE A NON ROOT USER:

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Note: the step ARG USER_GID=$USER_UID can take some minutes in the first setup (detected in CentOS 7 host)

# From here: non-root user (no sudo) -----------------------------------------------------------------------------------

# Instructions to Replicate in commands line. Valid for any Debian/Ubuntu machine (on-prem, WSL, CLoud VMs ...):
#   - Remove RUN: e.g.: RUN mkdir .....   -> mkdir ....
#   - Replace ENV by export: e.g.: ENV PATH="$HOME/.local/bin:$PATH"   -> export PATH="$HOME/.local/bin:$PATH"
# Note: Restating the shell to apply changes is sometimes required.

SHELL ["/usr/bin/bash", "-c"]
ENV SHELL /usr/bin/bash


# INSTALL MINICONDA (user)
RUN mkdir -p ~/miniconda3 && \
  curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda3/miniconda.sh && \
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
  rm ~/miniconda3/miniconda.sh && \
  ~/miniconda3/bin/conda init bash && \
  ~/miniconda3/bin/conda init zsh

# RUN /usr/bin/bash
# RUN source ~/.bashrc
# Allow git to work in any directory as root (security issue but useful for dev containers)
# RUN git config --global --add safe.directory '*'

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$HOME/.local/bin:$PATH"
# RUN source ~/.bashrc

# Git: Change default branch to main
RUN git config --global init.defaultBranch main
RUN source ~/.bashrc

# Create a Customized conda dev environment with cookiecutter and ipykernel  (last step to avoid env conflicts):
#  Do not use ENV PATH=~/miniconda3/bin:$PATH for conda before another block of commands
ENV PATH=~/miniconda3/bin:$PATH
RUN conda update --all -y && conda create -n dev python=3.10 -y && conda init bash && \
  echo "conda activate dev" >> ~/.bashrc && source activate dev && \
  conda install -y -c conda-forge 'cookiecutter>=2' ipykernel

# Entrypoint - BASH
CMD [ "/usr/bin/bash"]
