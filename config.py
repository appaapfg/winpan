"""Manages configuration for the Panopticon application.

This module determines the paths for critical configuration and data files,
including settings, detection models, and category mappings. It employs a
fallback system to prioritize local files (for development or portable use)
before defaulting to standard user-specific configuration and data directories.
If required configuration files are missing, it triggers a PySide6 error dialog
instructing the user to place them in the user configuration directory.

Attributes:
    APP_NAME (str): The name of the application, used for user directories.
    SETTINGS_FILE (str): The absolute path to the main settings YAML file.
    DETECTION_MODELS_FILE (str): The absolute path to the detection models YAML file.
    CATEGORIES_FILE (str): The absolute path to the COCO category JSON file.
    MODEL_DIRECTORY (str): The directory where ML models are stored.
    VERSION (str): The application's version number.
    id2label (dict): A mapping from category IDs to human-readable label names, derived from CATEGORIES_FILE.
"""

import json
import os
import sys
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

# Define the application name
APP_NAME = "Panopticon"


# Default settings file in the current directory
local_settings_file = Path("./panopticon_settings.yaml")
local_detection_models_file = Path("./detection_models.yaml")
local_categories_file = Path("./panopticon_categories.coco.json")

if local_settings_file.is_file():
    SETTINGS_FILE = str(local_settings_file)
else:
    config_dir = Path(user_config_dir(APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE = str(config_dir / "panopticon_settings.yaml")

config_dir = Path(user_config_dir(APP_NAME))
user_detection_models_file = config_dir / "detection_models.yaml"

if local_detection_models_file.is_file():
    DETECTION_MODELS_FILE = str(local_detection_models_file)
elif user_detection_models_file.is_file():
    DETECTION_MODELS_FILE = str(user_detection_models_file)
else:
    config_dir.mkdir(parents=True, exist_ok=True)
    DETECTION_MODELS_FILE = str(user_detection_models_file)

user_categories_file = config_dir / "panopticon_categories.coco.json"

if local_categories_file.is_file():
    CATEGORIES_FILE = str(local_categories_file)
elif user_categories_file.is_file():
    CATEGORIES_FILE = str(user_categories_file)
else:
    config_dir.mkdir(parents=True, exist_ok=True)
    CATEGORIES_FILE = str(user_categories_file)


def _show_missing_config_error(missing_files):
    if not missing_files:
        return

    target_dir = str(config_dir)
    missing_list = "\n".join(f"- {name}" for name in missing_files)
    message = (
        "Required configuration files are missing.\n"
        f"Place these files in {target_dir}:\n"
        f"{missing_list}\n"
        "Restart Panopticon after adding them."
    )

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        owns_app = False
        if app is None:
            app = QApplication([])
            owns_app = True

        QMessageBox.critical(None, "Panopticon Configuration Missing", message)

        if owns_app:
            app.quit()
    except Exception:
        print(message, file=sys.stderr)


missing_configs = []
if not Path(SETTINGS_FILE).is_file():
    missing_configs.append("panopticon_settings.yaml")
if not Path(DETECTION_MODELS_FILE).is_file():
    missing_configs.append("detection_models.yaml")
if not Path(CATEGORIES_FILE).is_file():
    missing_configs.append("panopticon_categories.coco.json")

_show_missing_config_error(missing_configs)

# Model directory logic: prefer '../panopticon_models', else use OS-specific program data location
current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_model_dir = os.path.normpath(os.path.join(current_file_directory, "./panopticon_models"))
if os.path.isdir(relative_model_dir):
    MODEL_DIRECTORY = relative_model_dir
else:
    MODEL_DIRECTORY = os.path.join(user_data_dir(APP_NAME), "panopticon_models")
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

VERSION = "1.0.0"

# Build id2label mapping from categories file
id2label = {}
try:
    with open(CATEGORIES_FILE) as f:
        categories_data = json.load(f)
        for category in categories_data.get("categories", []):
            id2label[category["id"]] = category["name"]
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    # If categories file cannot be loaded, id2label will be empty
    pass
