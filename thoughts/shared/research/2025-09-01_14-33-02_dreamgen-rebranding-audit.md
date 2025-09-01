---
date: 2025-09-01T14:33:02Z
researcher: Claude Code
git_commit: 44fcfd243d12d7ee84e8bd602e067a0a4541f63b
branch: main
repository: dreamgen
topic: "DreamGen Rebranding Audit - CLI Command and Frontend Branding"
tags: [research, codebase, branding, cli, frontend, web-ui, rebranding]
status: complete
last_updated: 2025-09-01
last_updated_by: Claude Code
---

# Research: DreamGen Rebranding Audit - CLI Command and Frontend Branding

**Date**: 2025-09-01T14:33:02Z
**Researcher**: Claude Code
**Git Commit**: 44fcfd243d12d7ee84e8bd602e067a0a4541f63b
**Branch**: main
**Repository**: dreamgen

## Research Question
Audit the complete codebase to ensure proper rebranding to 'DreamGen' with 'dreamgen' as the CLI command, and verify frontend branding consistency.

## Summary
**Status**: ⚠️ **Partially Complete** - Core package properly rebranded, but significant inconsistencies remain

**Key Findings**:
- ✅ Package configuration correctly updated (`pyproject.toml`)
- ✅ CLI entry point properly configured (`dreamgen` command)
- ❌ CLI user-facing messages still reference "Continuous Image Generator"
- ❌ Web UI frontend still branded as "Continuous Image Generator"
- ❌ 50+ files contain old naming references requiring updates
- ❌ Docker, Kubernetes, and deployment configs use old names

## Detailed Findings

### ✅ Correctly Rebranded Components

#### Package Configuration (`pyproject.toml:2,69`)
- Package name: `"dreamgen"`
- CLI entry point: `dreamgen = "src.main:main"`
- Repository URLs: Updated to `https://github.com/killerapp/dreamgen`
- Description: "Generate unlimited AI images locally..."

#### README Documentation (`README.md:1,30,48`)
- Main title: "✨ DreamGen"
- Install command: `uv tool install dreamgen`
- CLI examples: `dreamgen generate`, `dreamgen loop`

### ❌ Frontend Branding Issues

#### Web UI Layout (`web-ui/app/layout.tsx:13,16`)
```typescript
title: "Continuous Image Generator — Agentic Insights"  // Should be "DreamGen"
openGraph: {
  title: "Continuous Image Generator — Agentic Insights"  // Should be "DreamGen"
}
```

#### Web UI Header (`web-ui/app/page.tsx:146,149`)
```typescript
<span className="font-semibold text-sm text-foreground hidden sm:inline">
  Continuous Image Generator  // Should be "DreamGen"
</span>
<span className="font-semibold text-sm text-foreground sm:hidden">
  CIG  // Should be "DG" or "DreamGen"
</span>
```

#### Web UI Package (`web-ui/package.json:2`)
```json
"name": "web-ui"  // Should be "dreamgen-web-ui"
```

### ❌ CLI Branding Issues

#### Version Display (`src/utils/cli.py:62`)
```python
"[bold green]Continuous Image Generator[/bold green]\n"  # Should be "DreamGen"
```

#### Command Help Text (`src/utils/cli.py:87,90`)
```python
"""
🎨 Continuous Image Generation System  # Should be "🎨 DreamGen"

Run `uv run imagegen generate` for CLI usage or `uv run imagegen web`  # Should be "dreamgen"
```

#### File Docstrings (`src/utils/cli.py:2`, `src/main.py:2`)
```python
"""
Command-line interface for the continuous image generation system.  # Should be "DreamGen system"
```

### ❌ Infrastructure & Deployment Issues

#### Docker Configuration (`docker-compose.yml:8,49,64`)
```yaml
container_name: imagegen-backend     # → dreamgen-backend
container_name: imagegen-frontend    # → dreamgen-frontend
container_name: imagegen-ollama      # → dreamgen-ollama
```

#### CSO Module Configuration (`cso-module.yaml:6,12,14`)
```yaml
name: continuous-image-gen                    # → dreamgen
app.kubernetes.io/name: continuous-image-gen  # → dreamgen
url: https://github.com/killerapp/continuous-image-gen  # → dreamgen
```

#### Container Images (`cso-deployment/chart/values.yaml:21,86`)
```yaml
repository: ghcr.io/killerapp/continuous-image-gen-backend   # → dreamgen-backend
repository: ghcr.io/killerapp/continuous-image-gen-frontend  # → dreamgen-frontend
```

### ❌ Documentation & Help References

#### Claude.md (`CLAUDE.md:10-22`)
```bash
# All examples still use old command:
uv run imagegen generate
uv run imagegen generate --interactive
uv run imagegen loop --batch-size 10 --interval 300
# Should be:
uv run dreamgen generate
```

## Code References
- `pyproject.toml:2,69` - Package name and CLI entry point (✅ Correct)
- `src/utils/cli.py:44,62,87,90` - CLI branding inconsistencies
- `web-ui/app/layout.tsx:13,16` - Web UI title and metadata
- `web-ui/app/page.tsx:146,149` - Web UI header display
- `docker-compose.yml:8,49,64,72,87,93,97` - Container naming
- `cso-module.yaml:6,12,14,20,30` - Kubernetes configuration
- `CLAUDE.md:10-22` - Documentation examples

## Architecture Insights

### Dual Web Architecture
The project maintains two web components:
1. **Next.js Web UI** (`web-ui/`) - Main frontend application
2. **Cloudflare Worker** (`host-image/`) - Image hosting service

### CLI Integration
- CLI properly configured with `dreamgen` entry point
- Web interface launched via FastAPI backend (`src/api/server.py`)
- Current help text references non-existent `web` command

## Priority Fixes Required

### High Priority (User-Facing)
1. **CLI branding** - Update version display and help text
2. **Web UI frontend** - Update title, header, and package name
3. **Documentation** - Update CLAUDE.md command examples

### Medium Priority (Infrastructure)
4. **Docker containers** - Update container and network names
5. **CSO deployment** - Update Kubernetes configurations
6. **Environment variables** - Update CSO_MODULE_NAME

### Low Priority (Internal)
7. **Monitoring dashboards** - Update Grafana/Prometheus references
8. **File docstrings** - Update internal documentation

## Recommendations

### Immediate Actions
1. Update CLI version display and help text (`src/utils/cli.py`)
2. Update web UI title and header (`web-ui/app/layout.tsx`, `web-ui/app/page.tsx`)
3. Fix documentation examples in `CLAUDE.md`

### Systematic Updates
1. Search and replace `imagegen` → `dreamgen` in CLI commands
2. Search and replace `Continuous Image Generator` → `DreamGen`
3. Update container and deployment configurations
4. Consider version alignment between main package (1.0.0) and web-ui (0.1.0)

## Open Questions
1. Should mobile abbreviation be "DG" or "Dream" instead of "CIG"?
2. Should web-ui package name include version alignment with main package?
3. Are there database/infrastructure names that need updating for consistency?

---

**Total files requiring updates**: ~50+ files across CLI, frontend, documentation, and deployment configurations.
