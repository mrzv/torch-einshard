#!/bin/sh

uv run torchrun --nproc-per-node 8 -m pytest
