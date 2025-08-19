#!/bin/bash
# Script to build and push Docker images to GitHub Container Registry for CSO deployment

set -e

# Configuration
REGISTRY="ghcr.io"
NAMESPACE="killerapp"
BACKEND_IMAGE="continuous-image-gen-backend"
FRONTEND_IMAGE="continuous-image-gen-frontend"
VERSION="${1:-1.0.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting CSO Module Image Build and Push${NC}"
echo -e "${YELLOW}Version: ${VERSION}${NC}"

# Check if logged in to GitHub Container Registry
echo -e "\n${YELLOW}Checking GitHub Container Registry authentication...${NC}"
if ! docker login ${REGISTRY} --username ${GITHUB_USERNAME} --password ${GITHUB_TOKEN} 2>/dev/null; then
    echo -e "${RED}❌ Failed to authenticate with GitHub Container Registry${NC}"
    echo "Please set GITHUB_USERNAME and GITHUB_TOKEN environment variables"
    exit 1
fi

# Build Backend Image
echo -e "\n${YELLOW}Building backend image...${NC}"
docker build -f Dockerfile.backend \
    --build-arg VERSION=${VERSION} \
    -t ${BACKEND_IMAGE}:${VERSION} \
    -t ${BACKEND_IMAGE}:latest \
    .

# Tag Backend Image for Registry
docker tag ${BACKEND_IMAGE}:${VERSION} ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:${VERSION}
docker tag ${BACKEND_IMAGE}:latest ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:latest

# Build Frontend Image
echo -e "\n${YELLOW}Building frontend image...${NC}"
docker build -f Dockerfile.frontend \
    --build-arg VERSION=${VERSION} \
    --build-arg NEXT_PUBLIC_API_URL=http://continuous-image-gen-backend:8000 \
    -t ${FRONTEND_IMAGE}:${VERSION} \
    -t ${FRONTEND_IMAGE}:latest \
    .

# Tag Frontend Image for Registry
docker tag ${FRONTEND_IMAGE}:${VERSION} ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:${VERSION}
docker tag ${FRONTEND_IMAGE}:latest ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:latest

# Push Backend Images
echo -e "\n${YELLOW}Pushing backend images to registry...${NC}"
docker push ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:${VERSION}
docker push ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:latest

# Push Frontend Images
echo -e "\n${YELLOW}Pushing frontend images to registry...${NC}"
docker push ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:${VERSION}
docker push ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:latest

# Verify Images
echo -e "\n${YELLOW}Verifying pushed images...${NC}"
docker manifest inspect ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:${VERSION} > /dev/null 2>&1 && \
    echo -e "${GREEN}✅ Backend image verified${NC}" || \
    echo -e "${RED}❌ Backend image verification failed${NC}"

docker manifest inspect ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:${VERSION} > /dev/null 2>&1 && \
    echo -e "${GREEN}✅ Frontend image verified${NC}" || \
    echo -e "${RED}❌ Frontend image verification failed${NC}"

echo -e "\n${GREEN}🎉 CSO Module images successfully pushed!${NC}"
echo -e "Backend: ${REGISTRY}/${NAMESPACE}/${BACKEND_IMAGE}:${VERSION}"
echo -e "Frontend: ${REGISTRY}/${NAMESPACE}/${FRONTEND_IMAGE}:${VERSION}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Update Helm values with the new image tags"
echo "2. Deploy using: helm install continuous-image-gen ./cso-deployment/chart"
echo "3. Or sync ArgoCD application for GitOps deployment"