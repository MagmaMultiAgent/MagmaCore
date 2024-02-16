[![Docker Publish](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml/badge.svg?branch=v0.0.2-AQ)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/docker-publish.yml)&nbsp;[![Pylint](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml/badge.svg?branch=v0.0.2-AQ)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pylint.yml)&nbsp;[![CodeQL Advanced](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/codeql.yml/badge.svg?branch=v0.0.2-AQ)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/github-code-scanning/codeql)&nbsp;[![PyTest](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pytest.yml/badge.svg?branch=v0.0.2-AQ)](https://github.com/Getlar/VigIL-Game-Validation/actions/workflows/pytest.yml)&nbsp;[![CodeCoverage](https://github.com/MagmaMultiAgent/MagmaCore/actions/workflows/codecov.yml/badge.svg?branch=v0.0.2-AQ)](https://github.com/MagmaMultiAgent/MagmaCore/actions/workflows/codecov.yml)&nbsp;[![codecov](https://codecov.io/gh/MagmaMultiAgent/MagmaCore/graph/badge.svg?branch=v0.0.2-AQtoken=HFF0U8RVUE)](https://codecov.io/gh/MagmaMultiAgent/MagmaCore)&nbsp;![GitHub issues](https://img.shields.io/github/issues/MagmaMultiAgent/MagmaCore)&nbsp;



# Monolithic CNN-based solution with Action Queues

In this system, we're essentially orchestrating a **monolithic solution** where a central brain controls all units and factories across the grid. This process is an upgrade to the one on the [CNN-Mixed branch](https://github.com/MagmaMultiAgent/MagmaCore/tree/v0.0.1-CNN-mixed). Testig results can be found on the project [Wiki](https://github.com/MagmaMultiAgent/MagmaCore/wiki).


# Getting Started

I highly recommend utilizing Docker to set up and manage the environment on your system. Alternatively, a binary installation is also a viable option, especially considering the excellent performance and reliability of the official [pip](https://pypi.org/project/luxai-s2/) package.

## Docker

For implementing this solution, it's essential to have Docker Desktop installed on your system. Detailed guides for setting up Docker Desktop on [Mac](https://docs.docker.com/desktop/install/mac-install/), [Windows](https://docs.docker.com/desktop/install/windows-install/), and [Linux](https://docs.docker.com/desktop/install/linux-install/) can be accessed through the official Docker website. 

### CPU

Once Docker is properly configured on your system to execute the environment using a **CPU**, you can proceed by using the provided **run script**.

```bash
bash ./docker_run.sh -d cpu -v latest
```

### GPU

To employ [JAX](https://github.com/google/jax) as the backend and execute the environment on a GPU device, follow the script below:

```bash
bash ./docker_run.sh -d gpu -v latest
```

### DevContainers

Efficiently develop and test code within a Visual Studio Container by cloning the project and using `Ctrl+Shift+P` in VS Code's command palette.

```bash
> Dev Containers: Reopen in Container
```

On a Mac, using a Dev Container can lead to problems due to image incompatibility with `ARM processors`. For Macs, it's better to utilize [dockerRun](https://github.com/Getlar/VigIL-Game-Validation/blob/main/docker_run.sh). If you're on an `x86-64 processor`, opt for the `VS Code` dev container.

## Binary

To create a conda environment and use it run:

```bash
conda env create -f envs/environment.yml
conda activate luxai_s2
```

To install **additional packages** required to train and run specific agents run the following commands:

#### Devtools:
```bash
apt update -y && apt upgrade -y && apt install -y build-essential && apt-get install -y manpages-dev # Required if dev tools are missing.
```
#### Base packages:
```bash
pip install envs/environment_GH.txt
```

#### Install JAX support: (Optional)
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/RoboEden/jux.git@dev
```

To test the existing implementation check out the [running docs](https://github.com/Getlar/VigIL-Game-Validation/blob/v0.0.2-AQ/src/README.MD).
