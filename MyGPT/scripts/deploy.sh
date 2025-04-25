#!/bin/bash
# Docker deployment script

echo "Starting deployment..."
docker-compose up --build -d
echo "Deployment complete!"
