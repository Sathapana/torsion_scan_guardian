# syntax=docker/dockerfile:1.6
#
# Torsion Scan Guardian — reproducible Linux image.
#
# Default image is CPU-only (matches the Phase 0–5 dev environment, ~3 GB).
# To build a CUDA-enabled image for GPU hosts (RunPod, Lambda, etc.), pass:
#     docker build --build-arg BASE=mambaorg/micromamba:cuda12.1-ubuntu22.04 -t guardian:gpu .
# and ensure `--gpus all` at runtime so the container sees the GPU.
#
# Usage:
#   docker build -t guardian:cpu .
#   docker run --rm -it -v "$PWD/runs:/app/runs" guardian:cpu \
#       python -m guardian.cli --config config/default.yaml --calibrate
#
# For GPU:
#   docker run --rm -it --gpus all -v "$PWD/runs:/app/runs" guardian:gpu \
#       python -m guardian.cli --config config/default.yaml --steps 5000 \
#           --checkpoints runs/finetune_sulf/.../*.model --online-finetune \
#           --seed-data-file data/seed/sulfanilamide_seed.xyz

ARG BASE=mambaorg/micromamba:1.5.8

FROM ${BASE}

# Root for apt installs, then drop back to the standard mamba user.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /app

# Install conda deps first — slow layer, cache it aggressively.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml ./
RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean -ay

# Activate the base env for subsequent RUN / CMD commands.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy the package + scripts + configs + tests + seed datasets.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src/        ./src/
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/    ./scripts/
COPY --chown=$MAMBA_USER:$MAMBA_USER config/     ./config/
COPY --chown=$MAMBA_USER:$MAMBA_USER tests/      ./tests/
COPY --chown=$MAMBA_USER:$MAMBA_USER data/       ./data/

# Editable install of the guardian package.
RUN pip install --no-cache-dir -e .

# Pre-download MACE-OFF small so the first inference doesn't pay the network round-trip.
# This caches into /home/$MAMBA_USER/.cache/mace inside the image (~7 MB).
RUN python -c "from mace.calculators import mace_off; mace_off(model='small', device='cpu')"

# Run pytest at build time as a smoke test (fails the build if anything is broken).
RUN python -m pytest -q tests/ -k "not test_ensemble_predict_h2o"

# Windows codepages aren't relevant in a Linux container, but harmless to set.
ENV PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# Default command shows the CLI help — override at `docker run` to launch an experiment.
CMD ["python", "-m", "guardian.cli", "--help"]
