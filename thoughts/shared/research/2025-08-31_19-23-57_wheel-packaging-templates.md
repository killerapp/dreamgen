---
date: 2025-08-31T19:23:57-05:00
researcher: killerapp
git_commit: 740c254224118c232c5d897a6a707057489d4f1f
branch: main
repository: continuous-image-gen
topic: "Wheel packaging and template distribution investigation"
tags: [research, codebase, packaging, templates, wheel, cookiecutter, init]
status: complete
last_updated: 2025-08-31
last_updated_by: killerapp
---

# Research: Wheel Packaging and Template Distribution Investigation

**Date**: 2025-08-31T19:23:57-05:00
**Researcher**: killerapp
**Git Commit**: 740c254224118c232c5d897a6a707057489d4f1f
**Branch**: main
**Repository**: continuous-image-gen

## Research Question
Investigate what's being packaged in the wheels - specifically whether cookiecutter templates for agents and commands are included, and how the 'init' command installs templates (not actual 'thoughts' content but the thoughts-template that gets generated into a project's empty thoughts directory).

## Summary
The continuous-image-gen project currently packages only the `src` directory in its wheel distribution, missing critical data files required by plugins. No cookiecutter templates, init command, or thoughts-template directory exist in the codebase. This appears to be a standalone image generation application rather than a project scaffolding tool with template distribution capabilities.

## Detailed Findings

### Current Wheel Packaging Configuration

The project uses Hatchling as its build backend with minimal configuration:

**pyproject.toml Configuration** (`pyproject.toml:45-51`):
- Build backend: `hatchling.build`
- Packages included: Only `["src"]` directory
- Entry point: `imagegen = "src.main:main"`
- No MANIFEST.in file present
- No additional data files configured

**What Gets Packaged**:
- All Python modules in `src/` directory (25 files)
- Core components: generators, plugins, utils, API
- **Missing**: `data/` directory with critical JSON files

### Missing Critical Data Files

Two data files required by plugins are **not included** in the wheel:

1. **data/holidays.json** (`src/plugins/nearest_holiday.py:21`)
   - Required for holiday context in prompts
   - Plugin fails silently if missing

2. **data/art_styles.json** (`src/plugins/art_style.py:30`)
   - Contains 90+ art style definitions
   - Plugin logs errors if missing

### No Template Infrastructure Found

**No Cookiecutter Templates**:
- No `cookiecutter.json` files
- No cookiecutter template directories
- No `.jinja`, `.j2`, or `.tmpl` files
- No template generation code

**No Init Command**:
- CLI has only 3 commands: `generate`, `loop`, `diagnose` (`src/utils/cli.py`)
- No `init` or `initialize` command implementation
- No project scaffolding functionality
- Web command mentioned in docs but not implemented

**No Thoughts Template**:
- No `thoughts-template` directory exists
- No references to thoughts initialization in code
- Existing `thoughts/` directory is untracked and manually created
- No template system for generating thoughts structures

### Architecture Insights

The project follows these patterns for directory creation:
- **StorageManager** (`src/utils/storage.py:21`): Creates `output/YYYY/week_XX/` structure
- **Config paths** (`src/utils/config.py:158`): Uses `mkdir(parents=True, exist_ok=True)`
- **Logging** (`src/utils/logging_config.py:12`): Creates logs directory

However, none of these patterns apply to template distribution or project initialization.

## Code References
- `pyproject.toml:45-51` - Hatchling build configuration
- `src/utils/cli.py:15-150` - CLI command definitions (no init command)
- `src/plugins/nearest_holiday.py:21` - Requires missing holidays.json
- `src/plugins/art_style.py:30` - Requires missing art_styles.json
- `src/utils/storage.py:21` - Directory creation pattern example

## Conclusion

The continuous-image-gen project is a **standalone AI image generation system**, not a project scaffolding tool. It lacks:
1. Any template distribution system
2. An init command for project setup
3. Cookiecutter or similar template infrastructure
4. Thoughts-template for initializing project structures

To implement template distribution, you would need to:
1. Create the template directories and files
2. Configure Hatchling to include them in the wheel
3. Implement an init command in the CLI
4. Add template copying/generation logic

The current packaging also needs fixing to include the `data/` directory for plugins to function correctly.

## Open Questions
1. Is this project intended to become a scaffolding tool, or should it remain focused on image generation?
2. Should the missing `data/` files be included in the wheel distribution?
3. Is the unimplemented `web` command intentional or an oversight?