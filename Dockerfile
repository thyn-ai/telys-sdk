# Glama.ai build recipe for the telys MCP server (io.github.thyn-ai/telys).
#
# Builds an isolated uv venv with the PUBLIC telys package pinned to the current release
# and runs the stdio MCP server. The version pin below is bumped IN SYNC with every
# release by .github/workflows/cut-release.yml (it is part of the multi-file bump list)
# — do not hand-edit it between releases.
#
# initialize + tools/list introspection needs no credentials and no runtime (the tool
# list is static; the engine loads lazily per tool call). Tool calls that touch memory
# need the signed on-device runtime (`telys login`, then `telys runtime verify`).
FROM ghcr.io/astral-sh/uv:debian

RUN uv venv --seed /v \
 && /v/bin/pip install --no-cache telys==0.1.7 \
 && ln -sf /v/bin/telys /usr/local/bin/telys

CMD ["telys", "mcp"]
