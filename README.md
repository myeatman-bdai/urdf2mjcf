<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/urdf2mjcf/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/urdf2mjcf)](https://pypi.org/project/urdf2mjcf/)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/kscale)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/urdf2mjcf/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/urdf2mjcf/actions/workflows/test.yml)
[![Publish Python Package](https://github.com/kscalelabs/urdf2mjcf/actions/workflows/publish.yml/badge.svg)](https://github.com/kscalelabs/urdf2mjcf/actions/workflows/publish.yml)

</div>

# urdf2mjcf

![Example](./docs/example.png)

This script converts URDF files to MJCF files, with some nice options.

## Installation

```bash
pip install urdf2mjcf
```

## Usage

### Command Line

To run the conversion script from the command line, use:

```bash
urdf2mjcf path/to/your/robot.urdf
```

This will save the MJCF file in the same directory as the URDF file.

To see all the options, use:

```bash
urdf2mjcf -h
```

### Python

To run the conversion script from Python, use:

```python
from urdf2mjcf import run

run(
    urdf_path="path/to/your/robot.urdf",
    mjcf_path="path/to/save/robot.mjcf",
    copy_meshes=True,
)
```
