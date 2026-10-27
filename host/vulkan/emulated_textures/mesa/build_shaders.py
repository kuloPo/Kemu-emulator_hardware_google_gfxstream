#!/usr/bin/env python3
"""Build the Mesa ASTC storage-image variants using the Vulkan SDK glslc."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--glslc', default='glslc')
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / 'compiled'
output.mkdir(exist_ok=True)
for dimension in (2, 3):
    for hdr in (0, 1):
        profile = 'hdr' if hdr else 'ldr'
        destination = output / f'astc_{profile}_{dimension}d.inl'
        subprocess.run([args.glslc, '--target-env=vulkan1.1', '-fshader-stage=compute',
                        f'-DDIM={dimension}', f'-DASTC_HDR={hdr}', '-O', '-mfmt=num',
                        str(root / 'astc_decoder.comp'), '-o', str(destination)], check=True)
        print(destination.name)
