import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the given path.

    Args:
        config_path: Relative path to the configuration file from the project root.

    Returns:
        A dictionary containing the configuration.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config