
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu113.py310

WORKDIR /

COPY requirements.txt /

# Installs hypertune library
RUN pip install -r requirements.txt

# Copies the trainer code to the Docker image.
COPY processing /processing
