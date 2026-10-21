# Eleagent's day agent (agent/service.py): the naming image (archive reader, VLM helpers, graph driver) plus the
# OpenAI Agents SDK, installed against the image's own pins. Build context: services/recognition.
ARG AGENT_BASE_IMAGE=local/eleagent-naming:production
FROM ${AGENT_BASE_IMAGE}
RUN pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/base-constraints.txt \
    && pip install --no-cache-dir -c /tmp/base-constraints.txt openai-agents==0.24.0 \
    && python -c "import agents, openai, mcp; print('openai', openai.__version__)" \
    && rm /tmp/base-constraints.txt
COPY agent ./agent
CMD ["-m", "agent.service"]
