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
print('==============13==============', flush=True)
import json
import os
import sys
from pathlib import Path
print('==============14==============', flush=True)
from platformdirs import user_config_dir, user_data_dir
print('==============15==============', flush=True)
# Define the application name
APP_NAME = "Panopticon"


# Default settings file in the current directory
local_settings_file = Path("./panopticon_settings.yaml")
local_detection_models_file = Path("./detection_models.yaml")
local_categories_file = Path("./panopticon_categories.coco.json")
print('==============16==============', flush=True)
if local_settings_file.is_file():
    SETTINGS_FILE = str(local_settings_file)
else:
    config_dir = Path(user_config_dir(APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE = str(config_dir / "panopticon_settings.yaml")
print('==============17==============', flush=True)
config_dir = Path(user_config_dir(APP_NAME))
user_detection_models_file = config_dir / "detection_models.yaml"
print('==============18==============', flush=True)
if local_detection_models_file.is_file():
    DETECTION_MODELS_FILE = str(local_detection_models_file)
elif user_detection_models_file.is_file():
    DETECTION_MODELS_FILE = str(user_detection_models_file)
else:
    config_dir.mkdir(parents=True, exist_ok=True)
    DETECTION_MODELS_FILE = str(user_detection_models_file)
print('==============19==============', flush=True)
user_categories_file = config_dir / "panopticon_categories.coco.json"

if local_categories_file.is_file():
    CATEGORIES_FILE = str(local_categories_file)
elif user_categories_file.is_file():
    CATEGORIES_FILE = str(user_categories_file)
else:
    config_dir.mkdir(parents=True, exist_ok=True)
    CATEGORIES_FILE = str(user_categories_file)

print('==============20==============', flush=True)
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
    print('==============21==============', flush=True)

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtCore import QTimer
        print('==============22==============', flush=True)

        app = QApplication.instance()
        print('==============22.1==============', flush=True)
        owns_app = False
        if app is None:
            app = QApplication([])
            print('==============22.2==============', flush=True)
            owns_app = True

        print('==============22.3==============', flush=True)
        #QMessageBox.critical(None, "Panopticon Configuration Missing", message)
        msg_to_display = QMessageBox(QMessageBox.Critical, "Panopticon Configuration Missing", message)
        QTimer.singleShot(3000, msg_to_display.accept)
        msg_to_display.exec()
        print('==============23==============', flush=True)

        if owns_app:
            app.quit()
        print('==============24==============', flush=True)
    except Exception:
        print(message, file=sys.stderr)

print('==============25==============', flush=True)
missing_configs = []
if not Path(SETTINGS_FILE).is_file():
    missing_configs.append("panopticon_settings.yaml")
if not Path(DETECTION_MODELS_FILE).is_file():
    missing_configs.append("detection_models.yaml")
if not Path(CATEGORIES_FILE).is_file():
    missing_configs.append("panopticon_categories.coco.json")
print('==============26==============', flush=True)
_show_missing_config_error(missing_configs)
print('==============27==============', flush=True)
# Model directory logic: prefer '../panopticon_models', else use OS-specific program data location
current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_model_dir = os.path.normpath(os.path.join(current_file_directory, "./panopticon_models"))
print('==============28==============', flush=True)
if os.path.isdir(relative_model_dir):
    MODEL_DIRECTORY = relative_model_dir
else:
    MODEL_DIRECTORY = os.path.join(user_data_dir(APP_NAME), "panopticon_models")
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
print('==============29==============', flush=True)
VERSION = "1.0.0"

# Build id2label mapping from categories file
id2label = {}
try:
    with open(CATEGORIES_FILE) as f:
        print('==============30==============', flush=True)
        categories_data = json.load(f)
        for category in categories_data.get("categories", []):
            id2label[category["id"]] = category["name"]
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    # If categories file cannot be loaded, id2label will be empty
    pass
print('==============31==============', flush=True)
