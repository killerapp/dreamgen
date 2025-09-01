# Restore Typer CLI Functionality Implementation Plan

## Overview
Fix the Typer-based CLI to restore critical template installation functionality lost during the migration from Click, focusing on cookiecutter integration while preserving the consolidated init command improvements.

## Current State Analysis
The migration from Click to Typer has resulted in:
- Template installation broken - init command uses smart setup but doesn't install cookiecutter templates (agents/commands)
- Commands consolidated correctly - `quick-start` and `bootstrap` functionality merged into `init` 
- Semantic search added as experimental feature (working correctly)
- Find command restructured with subcommands (working correctly)
- Template resources not accessible in installed packages despite correct wheel packaging

### Key Discoveries:
- Templates are correctly packaged in wheel at `mem8/templates/` - `pyproject.toml:67-68`
- Typer init uses "smart setup" instead of templates - `mem8/cli_typer.py:790`
- Legacy Click implementation has full functionality - `mem8/legacy/cli_click_legacy.py:125`
- Template access pattern exists but fails - `mem8/core/memory.py:276`

## Desired End State
After implementation, the mem8 CLI should:
- Support cookiecutter template installation for agents and commands
- Preserve the smart setup functionality as default behavior
- Install templates when appropriate (Claude Code projects)
- Work correctly both in development and when installed from wheel

### Verification Criteria:
- `mem8 init` detects Claude Code projects and installs templates
- `mem8 init --template claude-config` installs Claude Code templates explicitly
- `mem8 init --template thoughts-repo` installs thoughts templates explicitly
- Templates accessible via `resources.files(mem8.templates)` in installed package
- Smart setup continues to work for non-Claude projects

## What We're NOT Doing
- Removing the smart setup functionality (it's the improved default)
- Breaking existing Typer CLI structure
- Modifying the wheel packaging configuration (it's correct)
- Changing the template content or structure
- Reverting to Click framework
- Re-adding separate `quick-start` or `bootstrap` commands (functionality is consolidated)
- Changing the find command structure (subcommands are working)

## Implementation Approach
We'll enhance the init command to install cookiecutter templates when appropriate while preserving the smart setup as the primary functionality.

## Phase 1: Add Template Installation to Init Command

### Overview
Enhance the init command to install cookiecutter templates for Claude Code projects while keeping smart setup as the default behavior.

### Changes Required:

#### 1. Update Init Command in Typer CLI
**File**: `mem8/cli_typer.py`
**Changes**: Add optional template parameter and cookiecutter integration

```python
# Line 789: Update init command signature to add template option
@app.command()
def init(
    template: Optional[str] = typer.Option(
        None,
        "--template", "-t",
        help="Force specific template: claude-config, thoughts-repo, or full (default: auto-detect)",
    ),
    repos: Annotated[Optional[str], typer.Option(
        "--repos", help="Comma-separated list of repository paths to discover"
    )] = None,
    # ... keep existing parameters ...
):
    """Initialize mem8 workspace with intelligent defaults and guided setup."""
    
    # After line 821 (after detecting project context):
    # Auto-detect if templates should be installed
    should_install_templates = False
    template_type = template  # Use explicit template if provided
    
    if template:
        # User explicitly requested templates
        should_install_templates = True
    elif context['is_claude_code_project']:
        # Auto-detect Claude Code projects need templates
        should_install_templates = True
        template_type = "claude-config"  # Default for Claude projects
    
    # After line 862 (after setup_minimal_structure):
    if should_install_templates:
        console.print("📦 [cyan]Installing templates...[/cyan]")
        _install_templates(template_type or "full", force, verbose)

# Add new function for template installation
def _install_templates(template_type: str, force: bool, verbose: bool) -> None:
    """Install cookiecutter templates to the workspace."""
    from cookiecutter.main import cookiecutter
    from importlib import resources
    import mem8.templates
    
    # Resolve template paths
    try:
        template_base = resources.files(mem8.templates)
    except (ImportError, AttributeError):
        # Development fallback
        template_base = Path(__file__).parent.parent
    
    # Map template types to directories
    template_map = {
        "full": ["claude-dot-md-template", "shared-thoughts-template"],
        "claude-config": ["claude-dot-md-template"],
        "thoughts-repo": ["shared-thoughts-template"],
    }
    
    if template_type not in template_map:
        console.print(f"[red]Invalid template: {template_type}[/red]")
        return
    
    workspace_dir = Path.cwd()
    
    # Run cookiecutter for each template
    for template_name in template_map[template_type]:
        template_path = template_base / template_name
        
        # Check if target already exists
        target_dir = ".claude" if "claude" in template_name else "thoughts"
        if (workspace_dir / target_dir).exists() and not force:
            console.print(f"[yellow]Skipping {template_name} - {target_dir} already exists[/yellow]")
            continue
        
        try:
            # For Claude templates, ensure .claude directory is the output
            extra_context = {}
            if "claude" in template_name:
                extra_context = {"project_slug": ".claude"}
            
            cookiecutter(
                str(template_path),
                no_input=True,
                output_dir=str(workspace_dir),
                overwrite_if_exists=force,
                extra_context=extra_context
            )
            console.print(f"  ✓ Installed {template_name}")
        except Exception as e:
            if verbose:
                console.print(f"[yellow]Could not install {template_name}: {e}[/yellow]")
```

#### 2. Add Helper Functions
**File**: `mem8/cli_typer.py`
**Changes**: Add conflict checking and backup functions

```python
def _check_conflicts(workspace_dir: Path, templates: List[str]) -> List[str]:
    """Check for existing files that would be overwritten."""
    conflicts = []
    
    for template in templates:
        if "claude" in template and (workspace_dir / ".claude").exists():
            conflicts.append(".claude directory")
        if "thoughts" in template and (workspace_dir / "thoughts").exists():
            conflicts.append("thoughts directory")
    
    return conflicts

def _backup_shared_thoughts(workspace_dir: Path) -> Optional[Path]:
    """Backup existing thoughts/shared directory."""
    shared_dir = workspace_dir / "thoughts" / "shared"
    if shared_dir.exists() and any(shared_dir.iterdir()):
        backup_dir = workspace_dir / ".mem8_backup" / "thoughts_shared"
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copytree(shared_dir, backup_dir, dirs_exist_ok=True)
        console.print(f"[yellow]Backed up thoughts/shared to {backup_dir}[/yellow]")
        return backup_dir
    
    return None

def _restore_shared_thoughts(workspace_dir: Path, backup_dir: Path) -> None:
    """Restore backed up thoughts/shared directory."""
    if backup_dir.exists():
        import shutil
        shared_dir = workspace_dir / "thoughts" / "shared"
        shutil.copytree(backup_dir, shared_dir, dirs_exist_ok=True)
        console.print("[green]Restored thoughts/shared from backup[/green]")
        shutil.rmtree(backup_dir.parent)
```

### Success Criteria:

#### Automated Verification:
- [x] Templates resolve correctly: `python -c "from importlib import resources; import mem8.templates; print(resources.files(mem8.templates))"`
- [ ] Unit tests pass: `uv run pytest tests/test_cli.py::test_init`
- [x] Cookiecutter imports successfully: `python -c "from cookiecutter.main import cookiecutter"`
- [ ] Template files exist in package: `python -m zipfile -l dist/*.whl | grep templates`

#### Manual Verification:
- [ ] `mem8 init` in Claude Code project auto-installs templates
- [x] `mem8 init --template full` creates both .claude and thoughts directories
- [x] `mem8 init --template claude-config` creates only .claude directory with agents/commands
- [x] `mem8 init --template thoughts-repo` creates only thoughts directory
- [x] Smart setup continues to work for non-Claude projects

---

## Phase 2: Ensure Template Resources Are Accessible

### Overview
Verify and fix template resource access in installed packages.

### Changes Required:

#### 1. Verify Templates Directory
**File**: `mem8/templates/__init__.py`
**Changes**: Ensure proper package initialization

```python
"""Template resources for mem8."""

# Ensure templates are discoverable
__all__ = ["claude-dot-md-template", "shared-thoughts-template"]
```

#### 2. Add Template Verification
**File**: `mem8/core/utils.py`
**Changes**: Add template verification utility

```python
def verify_templates() -> bool:
    """Verify that template resources are accessible."""
    try:
        from importlib import resources
        import mem8.templates
        
        template_base = resources.files(mem8.templates)
        
        # Check for both templates
        claude_template = template_base / "claude-dot-md-template"
        thoughts_template = template_base / "shared-thoughts-template"
        
        # Try to list files in templates
        claude_exists = (claude_template / "cookiecutter.json").exists()
        thoughts_exists = (thoughts_template / "cookiecutter.json").exists()
        
        return claude_exists and thoughts_exists
        
    except Exception:
        return False

def get_template_path(template_name: str) -> Path:
    """Get the path to a template, with fallback to development."""
    try:
        from importlib import resources
        import mem8.templates
        
        template_path = resources.files(mem8.templates) / template_name
        if template_path.exists():
            return template_path
    except Exception:
        pass
    
    # Development fallback
    dev_path = Path(__file__).parent.parent.parent / template_name
    if dev_path.exists():
        return dev_path
    
    raise FileNotFoundError(f"Template not found: {template_name}")
```

### Success Criteria:

#### Automated Verification:
- [ ] Build wheel: `uv build`
- [ ] Install wheel: `uv pip install dist/mem8-*.whl`
- [x] Verify templates: `python -c "from mem8.core.utils import verify_templates; print(verify_templates())"`
- [x] Templates accessible: `python -c "from mem8.core.utils import get_template_path; print(get_template_path('claude-dot-md-template'))"`

#### Manual Verification:
- [ ] Install from wheel and run `mem8 init` successfully
- [x] Templates are copied correctly to workspace
- [x] Both development and installed package scenarios work

---

## Testing Strategy

### Unit Tests:
- Test template resolution with mocked resources
- Test auto-detection of Claude Code projects
- Test template installation logic
- Test smart setup preservation

### Integration Tests:
- Test full init flow with actual templates
- Test Claude project detection and template installation
- Test explicit template selection
- Test template accessibility in wheel

### Manual Testing Steps:
1. Build wheel: `uv build`
2. Install in fresh environment: `uv pip install dist/mem8-*.whl`
3. Run `mem8 init` in Claude Code project (should auto-install templates)
4. Run `mem8 init --template full` in empty directory
5. Verify .claude and thoughts directories created with proper content
6. Test that agents and commands are properly installed

## Performance Considerations
- Template copying is I/O bound - consider caching template paths
- Resource resolution should be cached after first access
- Smart setup detection remains efficient

## Migration Notes
- Existing users get enhanced functionality automatically
- Smart setup remains the default behavior
- Template installation is additive, not replacing smart setup
- Legacy Click CLI remains available as fallback

## References
- Original research: `thoughts/shared/research/2025-09-01_wheel-packaging-templates.md`
- Legacy implementation: `mem8/legacy/cli_click_legacy.py:125`
- Current broken implementation: `mem8/cli_typer.py:790`
- Template access pattern: `mem8/core/memory.py:276`