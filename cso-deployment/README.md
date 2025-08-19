# CloudStack Orchestrator (CSO) Module Deployment

This directory contains the complete CSO module deployment configuration for the Continuous Image Generator system.

## Overview

The Continuous Image Generator is deployed as a CSO module following enterprise GitOps patterns with:
- Full Kubernetes/Helm deployment
- Istio service mesh integration
- OAuth2/Keycloak authentication
- Prometheus monitoring & Grafana dashboards
- ArgoCD GitOps deployment
- Multi-environment support

## Directory Structure

```
cso-deployment/
├── chart/                      # Helm chart
│   ├── Chart.yaml             # Chart metadata
│   ├── values.yaml            # Default values
│   ├── values-production.yaml # Production overrides
│   └── templates/             # Kubernetes manifests
├── monitoring/
│   ├── dashboard.json         # Grafana dashboard
│   └── alerts.yaml           # Prometheus alerts
├── argocd/
│   └── application.yaml      # ArgoCD application
└── push-images.sh           # Docker image push script
```

## Prerequisites

1. **Kubernetes Cluster** (1.24+)
2. **CloudStack Orchestrator Platform** installed with:
   - ArgoCD
   - Istio Service Mesh
   - Keycloak
   - Prometheus & Grafana
   - External Secrets Operator
3. **Docker Registry Access** (GitHub Container Registry)
4. **Helm** (3.10+)

## Quick Start

### 1. Build and Push Docker Images

```bash
# Set GitHub credentials
export GITHUB_USERNAME=your-username
export GITHUB_TOKEN=your-token

# Build and push images (version 1.0.0)
./cso-deployment/push-images.sh 1.0.0
```

### 2. Deploy with Helm

```bash
# Create namespace
kubectl create namespace continuous-image-gen

# Install the module
helm install continuous-image-gen ./cso-deployment/chart \
  --namespace continuous-image-gen \
  --values ./cso-deployment/chart/values.yaml

# For production
helm install continuous-image-gen ./cso-deployment/chart \
  --namespace continuous-image-gen \
  --values ./cso-deployment/chart/values.yaml \
  --values ./cso-deployment/chart/values-production.yaml
```

### 3. Deploy with ArgoCD (GitOps)

```bash
# Apply ArgoCD application
kubectl apply -f ./cso-deployment/argocd/application.yaml

# Sync the application
argocd app sync continuous-image-gen
```

## Configuration

### Environment Variables

Key environment variables configured in `values.yaml`:

- `USE_MOCK_GENERATOR`: Use mock image generation (no GPU required)
- `HF_TOKEN`: Hugging Face token for model downloads
- `OLLAMA_HOST`: Ollama service endpoint
- `LOG_LEVEL`: Logging verbosity

### Storage

The module requires two PersistentVolumeClaims:
- **Models**: 100Gi for AI models
- **Output**: 50Gi for generated images

### Authentication

OAuth2 Proxy + Keycloak integration:
- Configure in `values.yaml` under `auth.oauth2Proxy`
- Set client ID and secret via External Secrets

### Networking

- Ingress via Istio Gateway
- mTLS enabled between services
- Network policies for security

## Monitoring

### Grafana Dashboard

Import the dashboard from `monitoring/dashboard.json`:
- Image generation rate
- Error rates
- Latency metrics
- Resource usage

### Alerts

Prometheus alerts configured in `monitoring/alerts.yaml`:
- High error rate
- Slow generation
- Memory/CPU pressure
- Pod restarts
- Storage usage

## Multi-Environment Deployment

### Development
```bash
helm install continuous-image-gen ./cso-deployment/chart \
  --set global.cso.environment=development
```

### Staging
```bash
helm install continuous-image-gen ./cso-deployment/chart \
  --set global.cso.environment=staging
```

### Production
```bash
helm install continuous-image-gen ./cso-deployment/chart \
  --values ./cso-deployment/chart/values-production.yaml \
  --set global.cso.environment=production
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -n continuous-image-gen
kubectl describe pod <pod-name> -n continuous-image-gen
```

### View Logs
```bash
kubectl logs -n continuous-image-gen deployment/continuous-image-gen-backend
kubectl logs -n continuous-image-gen deployment/continuous-image-gen-frontend
```

### Test Endpoints
```bash
# Backend health
curl https://image-gen.example.com/api/health

# Frontend
curl https://image-gen.example.com/
```

### ArgoCD Issues
```bash
argocd app get continuous-image-gen
argocd app logs continuous-image-gen
```

## Security Considerations

1. **Pod Security**:
   - Runs as non-root user (UID 1000)
   - Minimal capabilities
   - Security contexts enforced

2. **Network Security**:
   - Network policies enabled
   - mTLS via Istio
   - Ingress with TLS termination

3. **Secrets Management**:
   - External Secrets Operator
   - Sealed Secrets support
   - No hardcoded credentials

## Backup & Recovery

### Backup Generated Images
```bash
kubectl exec -n continuous-image-gen deployment/continuous-image-gen-backend -- \
  tar czf /tmp/backup.tar.gz /app/output

kubectl cp continuous-image-gen/<pod>:/tmp/backup.tar.gz ./backup.tar.gz
```

### Restore
```bash
kubectl cp ./backup.tar.gz continuous-image-gen/<pod>:/tmp/backup.tar.gz

kubectl exec -n continuous-image-gen deployment/continuous-image-gen-backend -- \
  tar xzf /tmp/backup.tar.gz -C /
```

## Integration with CSO Platform

This module integrates with the CloudStack Orchestrator platform:

1. **Service Mesh**: Automatic sidecar injection
2. **Authentication**: Keycloak SSO
3. **Monitoring**: Prometheus ServiceMonitor
4. **GitOps**: ArgoCD application
5. **Secrets**: External Secrets Operator

## Support

For issues or questions:
- GitHub Issues: https://github.com/killerapp/continuous-image-gen/issues
- Documentation: https://docs.agenticinsights.com/cso/modules
- Email: support@killerapp.com