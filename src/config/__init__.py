import os

import yaml


def get_config():
    project_dir = "."  # os.path.dirname(os.path.dirname(os.path.abspath("")))
    path = os.path.join(project_dir, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
