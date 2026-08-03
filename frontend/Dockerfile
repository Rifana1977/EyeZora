# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install deps and build the Next.js app
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files first for better layer caching.
COPY package.json package-lock.json ./
RUN npm config set fetch-retries 5 && npm ci

# Copy the rest of the source code
COPY . .

# Receive the public API URL at build time.
ARG NEXT_PUBLIC_API_URL=http://localhost:5000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Build the standalone Next.js production bundle
RUN npm run build

# ── Stage 2: Runner ───────────────────────────────────────────────────────────
# Minimal production image — only the standalone output is needed
FROM node:20-slim AS runner

WORKDIR /app

ENV NODE_ENV=production

# Create a non-root user for security
RUN groupadd --system --gid 1001 nodejs && \
    useradd  --system --uid 1001 nextjs

# Copy the standalone build output from the builder stage.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static    ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public          ./public

USER nextjs

EXPOSE 3000

# The standalone server entry point produced by Next.js
CMD ["node", "server.js"]
