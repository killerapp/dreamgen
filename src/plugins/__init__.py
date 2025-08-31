"""Plugin system for adding contextual information to prompts."""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict

from ..utils.config import Config
from ..utils.plugin_manager import PluginManager

# Global plugin manager instance used across the application
plugin_manager = PluginManager()
logger = logging.getLogger(__name__)
_initialized = False


def initialize_plugins(config: Config) -> None:
    """Discover and register plugins dynamically based on configuration."""
    global _initialized
    plugin_manager.plugins.clear()

    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")

        if hasattr(module, "get_plugin"):
            name, description, func = module.get_plugin(config)
            enabled = name in config.plugins.enabled_plugins
            order = config.plugins.plugin_order.get(name, 999)
            plugin_manager.register(name, description, func, enabled=enabled, order=order)
            logger.debug(f"Discovered plugin: {name}")

    _initialized = True


def ensure_initialized(config: Config) -> None:
    """Initialize plugins on first use."""
    if not _initialized or not plugin_manager.plugins:
        initialize_plugins(config)


def get_context_with_descriptions(config: Config) -> Dict[str, Any]:
    """Execute plugins and return their results with descriptions."""
    ensure_initialized(config)
    results = plugin_manager.execute_plugins()

    for result in results:
        logger.info(f"Plugin contribution - {result.name}: {result.value} ({result.description})")

    return {
        "results": results,
        "descriptions": plugin_manager.get_plugin_descriptions(),
    }


def get_temporal_descriptor(config: Config) -> str:
    """Create a human-readable string combining plugin contributions."""
    ensure_initialized(config)
    results = plugin_manager.execute_plugins()

    parts = []
    holiday_fact = None
    art_style = None

    for result in results:
        if result.name == "holiday_fact":
            holiday_fact = result.value
        elif result.name == "art_style":
            art_style = result.value
        else:
            if result.value:
                parts.append(str(result.value))

    descriptor = ", ".join(filter(None, parts))

    if holiday_fact:
        descriptor = f"{descriptor} ({holiday_fact})"

    if art_style:
        descriptor = f"{descriptor}, {art_style}"

    logger.info(f"Generated temporal descriptor: {descriptor}")
    return descriptor
