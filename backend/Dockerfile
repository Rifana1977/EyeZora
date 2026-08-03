# ── EyeZora Backend (Node.js + Express) ──────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Install deps first (cached layer — only invalidated if package files change)
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy the application source
COPY . .

# Ensure runtime-writable directories exist inside the container.
# exam_logs and temp_uploads are written at runtime by the Express app.
RUN mkdir -p exam_logs temp_uploads

# Run as non-root for security
# node:20-alpine ships with a built-in 'node' user (uid 1000)
RUN chown -R node:node /app
USER node

EXPOSE 5000

CMD ["node", "server.js"]
