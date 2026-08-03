# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install deps and build the Next.js app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for better layer caching.
# Docker will reuse the cached npm ci layer if only source code changes.
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Receive the public API URL at build time.
# NEXT_PUBLIC_* vars are baked into the JS bundle during `next build`.
ARG NEXT_PUBLIC_API_URL=http://localhost:5000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Build the standalone Next.js production bundle
RUN npm run build

# ── Stage 2: Runner ───────────────────────────────────────────────────────────
# Minimal production image — only the standalone output is needed
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Create a non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

# Copy the standalone build output from the builder stage.
# standalone/ contains its own minimal node_modules (no full copy needed).
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static    ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public          ./public

USER nextjs

EXPOSE 3000

# The standalone server entry point produced by Next.js
CMD ["node", "server.js"]
