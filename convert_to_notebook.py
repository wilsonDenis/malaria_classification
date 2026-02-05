#!/usr/bin/env python3

import subprocess
import sys

print("Conversion du script Python en notebook Jupyter...")

script_content = open('classification_pytorch.py', 'r').read()

cells = []
current_cell = []
cell_type = 'code'

for line in script_content.split('\n'):
    if line.strip().startswith('#') and '=' in line:
        if current_cell:
            cells.append(('code', '\n'.join(current_cell)))
            current_cell = []
        cells.append(('markdown', f"## {line.strip('#').strip()}"))
    else:
        current_cell.append(line)

if current_cell:
    cells.append(('code', '\n'.join(current_cell)))

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for cell_type, content in cells:
    if cell_type == 'markdown':
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [content]
        })
    else:
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.split('\n')
        })

with open('classification_malaria_complete.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Notebook créé: classification_malaria_complete.ipynb")
