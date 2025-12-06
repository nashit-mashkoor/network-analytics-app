#!/bin/bash

echo "🔄 Rebuilding container (no cache)..."
docker compose build --no-cache

echo "🚀 Starting container..."
docker compose up -d

echo "✅ Done! Container is running."
docker compose ps


