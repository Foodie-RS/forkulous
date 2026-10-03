# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# This Dockerfile uses Docker Hardened Images (DHI) for enhanced security.
# For more information, see https://docs.docker.com/dhi/

# Use the dev image to build and install dependencies.
FROM pytorch/manylinux2_28_aarch64-builder:cpu-aarch64@sha256:96d4e6b9d82d0a30aac06b3d30c190a7bf89d5e2dea7c82d429f6478b6d78661

WORKDIR /api

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt

#COPY . .

EXPOSE 8000

CMD ["fastapi", "run", "/api/rest_api.py", "--port", "8000"]
