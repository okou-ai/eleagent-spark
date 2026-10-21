FROM node:24-bookworm
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund
COPY graph.mjs server.mjs main.mjs mcp.mjs validation.mjs archive.mjs auth.mjs auth-http.mjs auth-pages.mjs capture-proxy.mjs init-owner.mjs notices.mjs okou.mjs ./
USER node
CMD ["node", "main.mjs"]
