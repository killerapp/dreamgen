# CSO Bootstrap Enhancement Proposal

## Overview
This proposal enhances CloudStack Orchestrator's bootstrap capabilities to support automated module provisioning, particularly for modules requiring cloud resources like the continuous-image-gen module.

## Proposed Changes to CSO Roadmap

### Add to Phase 1.2: Module Development Experience

#### Module Bootstrap Framework
- [ ] **Automated Resource Provisioning**
  - Cloud resource templates (GCS, S3, Cloud SQL, etc.)
  - Terraform/Pulumi integration for infrastructure
  - Resource dependency resolution
  - Cost estimation before provisioning

- [ ] **Bootstrap Configuration Spec**
  - Extend module spec with `bootstrap` section
  - Define required cloud resources declaratively
  - Support multiple cloud providers (GCP, AWS, Azure)
  - Environment-specific bootstrap configs

- [ ] **Interactive Bootstrap Wizard**
  ```bash
  cso module bootstrap <module-name>
  # Interactive prompts for:
  # - Cloud provider selection
  # - Project/Account configuration
  # - Resource naming and sizing
  # - Secret management setup
  # - Network configuration
  ```

### Proposed Module Spec Extension

```yaml
apiVersion: cso.platform/v1alpha1
kind: ModuleSpecification
metadata:
  name: continuous-image-gen
spec:
  # Existing spec sections...
  
  bootstrap:
    description: "Resources required before module deployment"
    cloud:
      gcp:
        enabled: true
        project:
          required: true
          description: "GCP Project ID for resources"
        resources:
          - type: gcs-bucket
            name: image-storage
            config:
              location: us-central1
              storageClass: STANDARD
              lifecycle:
                enabled: true
                rules:
                  - age: 30
                    action: SetStorageClass
                    storageClass: NEARLINE
          - type: service-account
            name: module-sa
            roles:
              - storage.objectAdmin
              - secretmanager.secretAccessor
          - type: cloud-sql
            name: metadata-db
            optional: true
            config:
              version: POSTGRES_14
              tier: db-f1-micro
              databases:
                - imagegen
      aws:
        enabled: false
        # AWS-specific configuration
      azure:
        enabled: false
        # Azure-specific configuration
    
    secrets:
      required:
        - name: huggingface-token
          type: external
          description: "Hugging Face API token for model access"
          provider: user-input
        - name: gcs-service-account
          type: generated
          description: "Auto-generated during bootstrap"
      
    validation:
      preChecks:
        - type: api-enabled
          apis:
            - storage.googleapis.com
            - secretmanager.googleapis.com
            - sqladmin.googleapis.com
        - type: permissions
          roles:
            - roles/storage.admin
            - roles/iam.serviceAccountAdmin
      postChecks:
        - type: connectivity
          endpoints:
            - gcs-bucket
            - cloud-sql
        - type: secrets-exist
          secrets:
            - huggingface-token
```

### CSO CLI Enhancements

#### New Bootstrap Commands
```bash
# Bootstrap a module with all dependencies
cso module bootstrap continuous-image-gen \
  --project gcp-project-id \
  --region us-central1 \
  --environment production

# Validate bootstrap requirements
cso module bootstrap validate continuous-image-gen

# Destroy bootstrapped resources
cso module bootstrap destroy continuous-image-gen \
  --confirm

# Export bootstrap configuration
cso module bootstrap export continuous-image-gen \
  --format terraform > infrastructure.tf
```

#### Bootstrap Status Command
```bash
cso module bootstrap status continuous-image-gen

# Output:
# ✅ GCS Bucket: cso-imagegen-project-123
# ✅ Service Account: continuous-image-gen@project-123.iam
# ✅ Cloud SQL: cso-imagegen-db (RUNNING)
# ✅ Secrets: 3/3 configured
# ⚠️ Workload Identity: Not configured
# ❌ DNS: imagegen.platform.domain.com not resolving
```

### Integration with Existing CSO Components

#### 1. ArgoCD Integration
- Pre-sync hooks to validate bootstrap
- Bootstrap status as Application health check
- Automated secret synchronization

#### 2. External Secrets Operator
- Auto-configure SecretStore for module
- Map bootstrap secrets to ESO resources
- Support for multiple secret backends

#### 3. Terraform Controller (Optional)
- Manage infrastructure as GitOps
- Bootstrap resources via CRDs
- Drift detection and remediation

### Bootstrap Templates

Create reusable templates for common patterns:

```yaml
# bootstrap-templates/ai-workload.yaml
apiVersion: cso.platform/v1alpha1
kind: BootstrapTemplate
metadata:
  name: ai-workload
spec:
  description: "Standard bootstrap for AI/ML workloads"
  resources:
    storage:
      - type: object-storage
        purpose: model-cache
        size: 100Gi
      - type: object-storage  
        purpose: output-data
        size: 500Gi
    compute:
      - type: gpu-node-pool
        optional: true
        accelerator: nvidia-t4
    database:
      - type: vector-db
        optional: true
        purpose: embeddings
```

### Multi-Tenancy Considerations

#### Tenant-Specific Bootstrap
```yaml
bootstrap:
  multiTenant:
    enabled: true
    isolation: namespace
    resourcePrefix: "${tenant}-${module}"
    quotas:
      storage: 100Gi
      compute: 4-cores
```

### Cost Management

#### Bootstrap Cost Estimation
```bash
cso module bootstrap estimate continuous-image-gen \
  --environment production

# Estimated Monthly Costs:
# GCS Bucket (100GB): $20.00
# Cloud SQL (micro): $15.00
# Service Account Keys: $0.00
# Total: ~$35.00/month
```

### Security Enhancements

#### Secret Injection During Bootstrap
- Vault integration for secret generation
- Automated RBAC setup
- Least privilege service accounts
- Encryption at rest configuration

### Observability

#### Bootstrap Metrics
- Time to bootstrap completion
- Resource provisioning success rate
- Cost per bootstrap operation
- Failed bootstrap reasons

### Documentation Requirements

1. **Bootstrap Guide per Module Type**
   - AI/ML workloads
   - Web applications
   - Data processing pipelines
   - Stateful services

2. **Cloud Provider Guides**
   - GCP bootstrap guide
   - AWS bootstrap guide
   - Azure bootstrap guide
   - On-premise bootstrap guide

3. **Troubleshooting Guide**
   - Common bootstrap failures
   - Permission issues
   - Quota errors
   - Network connectivity

## Implementation Timeline

### Phase 1 (Immediate)
- Module spec extension for bootstrap
- Basic CLI commands
- GCP support only

### Phase 2 (Q2 2025)  
- Interactive wizard
- Multi-cloud support
- Cost estimation

### Phase 3 (Q3 2025)
- Terraform controller integration
- Bootstrap templates
- Multi-tenant bootstrap

## Benefits

1. **Reduced Time to Deploy**: From hours to minutes
2. **Consistency**: Standardized resource provisioning
3. **Cost Control**: Upfront cost visibility
4. **Security**: Automated security best practices
5. **Developer Experience**: No cloud expertise required

## Migration Path

For existing modules:
1. Add bootstrap section to module spec
2. Run `cso module bootstrap import`
3. Validate imported configuration
4. Test with `cso module bootstrap validate`

## Success Metrics

- 90% of modules use bootstrap framework
- 75% reduction in deployment time
- Zero security misconfigurations
- 100% secret rotation compliance