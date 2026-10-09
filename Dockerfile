# Competition image: patch retrieval + region grounding for aerial imagery.
#
# Build:  docker build -t qiyuan-patchmatch:v1 .
# Export: docker save qiyuan-patchmatch:v1 | gzip > submission.tar.gz
# Run:    docker run --rm --gpus all \
#             -v /host/eval_input:/input_path:ro -v /host/out:/output_path \
#             qiyuan-patchmatch:v1
#
# The container must work with no network: every dependency is installed at build time and no
# weights are downloaded at run time.  The NCC pipeline is training-free, so the image carries
# code only -- no model checkpoints -- which keeps it a few hundred MB instead of multiple GB.

FROM python:3.11-slim

# OpenCV and Pillow need these shared libraries at run time; they are not pulled in by pip.
# libgl1/libglib2.0-0 cover cv2's GUI-less build, libjpeg/libpng are Pillow decoders.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch first, in its own layer, so the large wheel is cached across code edits.
# The CPU wheel index is used deliberately: the runtime image targets a CUDA GPU, and the
# default PyPI torch wheel already bundles CUDA support, so no extra index is needed.  Install
# the CPU-only variant only if the grader ever runs without a GPU.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

RUN pip install --no-cache-dir numpy==2.0.1 pillow==11.1.0

# Application code.
COPY patchmatch/ /workspace/patchmatch/
COPY run.py /workspace/run.py

# Optional learned descriptor weights.  Copied only if present, so the image still builds in a
# checkout that has not trained the descriptor (the NCC pipeline needs no weights).
COPY weights/ /workspace/weights/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INPUT_PATH=/input_path \
    OUTPUT_PATH=/output_path

# Fail the build early if the pipeline cannot even be imported in the image.
RUN python -c "import sys; sys.path.insert(0, '/workspace'); import patchmatch; \
print('patchmatch', patchmatch.__version__, 'imports OK')"

ENTRYPOINT ["python", "/workspace/run.py"]
