# Continuous Image Gen - CSO Module Bootstrap Guide

This guide provides step-by-step instructions for bootstrapping the continuous-image-gen module in CloudStack Orchestrator on GCP.

## Prerequisites

- GCP Project with billing enabled
- `gcloud` CLI installed and authenticated
- `kubectl` configured for your GKE cluster
- CSO platform deployed with:
  - ArgoCD
  - Keycloak
  - Istio service mesh
  - External Secrets Operator

## Module-Specific Bootstrap Requirements

### 1. GCP Resources Setup

#### 1.1 Create GCS Bucket for Images
```bash
# Set your project ID
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET_NAME="cso-imagegen-${GCP_PROJECT_ID}"
export GCS_REGION="us-central1"

# Create bucket
gcloud storage buckets create gs://${GCS_BUCKET_NAME} \
  --project=${GCP_PROJECT_ID} \
  --location=${GCS_REGION} \
  --uniform-bucket-level-access

# Set lifecycle policy for cost optimization
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 30
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90
        }
      }
    ]
  }
}
EOF

gcloud storage buckets update gs://${GCS_BUCKET_NAME} \
  --lifecycle-file=lifecycle.json
```

#### 1.2 Create Service Account for GCS Access
```bash
# Create service account
export SA_NAME="continuous-image-gen"
gcloud iam service-accounts create ${SA_NAME} \
  --display-name="Continuous Image Gen Service Account" \
  --project=${GCP_PROJECT_ID}

# Grant storage permissions
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Create and download key
gcloud iam service-accounts keys create gcs-key.json \
  --iam-account=${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com
```

#### 1.3 Setup Cloud SQL Instance (Optional - for metadata storage)
```bash
# Create Cloud SQL instance
export INSTANCE_NAME="cso-imagegen-db"
export DB_REGION="us-central1"

gcloud sql instances create ${INSTANCE_NAME} \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=${DB_REGION} \
  --network=default \
  --no-assign-ip \
  --project=${GCP_PROJECT_ID}

# Create database
gcloud sql databases create imagegen \
  --instance=${INSTANCE_NAME} \
  --project=${GCP_PROJECT_ID}

# Create user
gcloud sql users create imagegen \
  --instance=${INSTANCE_NAME} \
  --password=<secure-password> \
  --project=${GCP_PROJECT_ID}
```

### 2. Secrets Configuration

#### 2.1 Store Secrets in GCP Secret Manager
```bash
# Store Hugging Face token
echo -n "your-huggingface-token" | gcloud secrets create huggingface-token \
  --data-file=- \
  --project=${GCP_PROJECT_ID}

# Store GCS service account key
gcloud secrets create gcs-service-account-key \
  --data-file=gcs-key.json \
  --project=${GCP_PROJECT_ID}

# Store database password (if using Cloud SQL)
echo -n "your-db-password" | gcloud secrets create imagegen-db-password \
  --data-file=- \
  --project=${GCP_PROJECT_ID}

# Grant access to GKE service account
export GKE_SA="continuous-image-gen@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
gcloud secrets add-iam-policy-binding huggingface-token \
  --member="serviceAccount:${GKE_SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --project=${GCP_PROJECT_ID}
```

#### 2.2 Configure External Secrets in Kubernetes
```yaml
# external-secret-store.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: gcpsm-imagegen
  namespace: continuous-image-gen
spec:
  provider:
    gcpsm:
      projectID: "your-project-id"
      auth:
        workloadIdentity:
          clusterLocation: us-central1
          clusterName: cso-cluster
          serviceAccountRef:
            name: continuous-image-gen
---
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: imagegen-secrets
  namespace: continuous-image-gen
spec:
  secretStoreRef:
    name: gcpsm-imagegen
    kind: SecretStore
  target:
    name: continuous-image-gen-secrets
  data:
    - secretKey: huggingface-token
      remoteRef:
        key: huggingface-token
    - secretKey: gcs-key
      remoteRef:
        key: gcs-service-account-key
    - secretKey: db-password
      remoteRef:
        key: imagegen-db-password
```

### 3. Keycloak Configuration

#### 3.1 Create Client in Keycloak
```bash
# Use CSO CLI or Keycloak admin UI
cso auth create-client \
  --name continuous-image-gen \
  --realm platform \
  --redirect-uri "https://imagegen.platform.your-domain.com/*" \
  --web-origins "https://imagegen.platform.your-domain.com"
```

#### 3.2 Create Roles
```bash
# Create module-specific roles
cso auth create-role --name image-generator-user --realm platform
cso auth create-role --name image-generator-admin --realm platform
```

### 4. Module Deployment

#### 4.1 Create Values File
```bash
# Copy template and fill in values
cp cso-deployment/values-template.yaml values.local.yaml

# Edit values.local.yaml with your configuration:
# - GCP project ID
# - GCS bucket name
# - Cloud SQL connection string
# - Domain configuration
```

#### 4.2 Deploy via ArgoCD
```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: continuous-image-gen
  namespace: argocd
  labels:
    cso.platform/module: continuous-image-gen
    cso.platform/tier: custom
spec:
  project: platform
  source:
    repoURL: https://github.com/killerapp/continuous-image-gen
    targetRevision: main
    path: cso-deployment/chart
    helm:
      valueFiles:
        - values.yaml
        - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: continuous-image-gen
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

#### 4.3 Apply ArgoCD Application
```bash
kubectl apply -f argocd-application.yaml
```

### 5. Ollama Integration (Optional)

If not already deployed, deploy Ollama for prompt generation:

```bash
# Deploy Ollama in the cluster
kubectl create namespace ollama-system
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: ollama-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        volumeMounts:
        - name: ollama-data
          mountPath: /root/.ollama
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-data
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
  namespace: ollama-system
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
EOF

# Pull required model
kubectl exec -n ollama-system deployment/ollama -- ollama pull llama2
```

### 6. Post-Deployment Verification

#### 6.1 Check Module Status
```bash
# Check if all pods are running
kubectl get pods -n continuous-image-gen

# Check ArgoCD sync status
argocd app get continuous-image-gen

# Verify Istio injection
kubectl get pods -n continuous-image-gen -o jsonpath='{.items[*].spec.containers[*].name}' | grep istio-proxy
```

#### 6.2 Test Access
```bash
# Get the ingress URL
export INGRESS_URL="https://imagegen.platform.your-domain.com"

# Test health endpoint
curl ${INGRESS_URL}/health

# Access web UI
open ${INGRESS_URL}
```

### 7. Module-Specific CSO Commands

Once deployed, use these CSO CLI commands:

```bash
# Validate module compliance
cso module validate continuous-image-gen

# Check module status
cso module status continuous-image-gen

# View module logs
cso module logs continuous-image-gen

# Scale module
cso module scale continuous-image-gen --replicas=3

# Update module configuration
cso module update continuous-image-gen --values values.local.yaml
```

## Troubleshooting

### Common Issues

1. **GCS Access Denied**
   - Verify service account has correct IAM roles
   - Check if Workload Identity is properly configured

2. **Model Download Fails**
   - Ensure Hugging Face token is valid
   - Check if pod has internet access (egress rules)

3. **Keycloak Authentication Issues**
   - Verify client configuration in Keycloak
   - Check redirect URIs match exactly

4. **Database Connection Failed**
   - Verify Cloud SQL proxy is running
   - Check connection string format
   - Ensure VPC peering is configured

## Security Considerations

1. **Secrets Rotation**
   - Rotate GCS service account keys quarterly
   - Update Hugging Face token if exposed
   - Use External Secrets Operator for automatic rotation

2. **Network Policies**
   - Module includes NetworkPolicies by default
   - Adjust egress rules as needed for your environment

3. **Image Scanning**
   - Container images are scanned with Trivy
   - Review scan results before production deployment

## Cost Optimization

1. **GCS Lifecycle Policies**
   - Images automatically move to cheaper storage classes
   - Adjust lifecycle rules based on access patterns

2. **Autoscaling**
   - HPA configured to scale based on CPU/memory
   - Adjust thresholds based on workload

3. **GPU/TPU Usage**
   - Only enable if needed for performance
   - Consider spot/preemptible instances for cost savings

## Integration with CSO Platform

This module integrates with:
- **Keycloak**: For authentication
- **Istio**: For service mesh and traffic management
- **Prometheus**: For metrics collection
- **Grafana**: For visualization (dashboard included)
- **ArgoCD**: For GitOps deployment
- **External Secrets**: For secret management

## Next Steps

1. Configure monitoring dashboards
2. Set up alerts for generation failures
3. Implement backup strategy for generated images
4. Configure CI/CD pipeline for module updates
5. Document custom plugin development