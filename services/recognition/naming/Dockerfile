# Spoken naming service (naming/service.py): the recognition image unchanged (the worker's download/decode path,
# DINOv2-S weights loader, ffmpeg/cv2 versions) plus the graph driver.
# Build context: services/recognition. The VLM is reached over HTTP (stdlib urllib); no extra client library.
ARG NAMING_BASE_IMAGE=local/eleagent-recognition:production
FROM ${NAMING_BASE_IMAGE}
RUN pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/base-constraints.txt \
    && pip install --no-cache-dir -c /tmp/base-constraints.txt neo4j==6.4.0 \
    && python -c "import neo4j; print(neo4j.__version__)" \
    && rm /tmp/base-constraints.txt
COPY naming ./naming
ENV OMP_NUM_THREADS=2
CMD ["-m", "naming.service"]
