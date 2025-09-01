#!/usr/bin/env python3
"""
Format code using black and isort
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Format the codebase"""
    repo_root = Path(__file__).parent.parent

    commands = [
        (["uv", "run", "black", "src/", "tests/"], "Formatting with black"),
        (["uv", "run", "isort", "src/", "tests/"], "Sorting imports with isort"),
    ]

    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False

    if success:
        print("✅ All formatting completed successfully!")
    else:
        print("❌ Some formatting commands failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
