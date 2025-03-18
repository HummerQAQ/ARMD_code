#!/usr/bin/env python3
import os
import sys

import toml


def replace_strings_in_file(fpath: str, old: str, replacement: str) -> None:
    with open(fpath) as f:
        s = f.read()
    s = s.replace(old, replacement)
    with open(fpath, "w") as f:
        f.write(s)


if __name__ == "__main__":
    old = "exodus_ts_model_template"
    if len(sys.argv) != 2:
        print("Need exactly one name for the model")
        sys.exit(1)
    replacement = sys.argv[1]
    if not os.path.exists(old):
        # NO OP
        with open("./pyproject.toml") as f:
            name = toml.load(f)["tool"]["poetry"]["name"]
        print(f'Found model algorithm "{name}", will not rename')
        sys.exit(0)
    targets = ["exodus_ts_model_template", "tests"]
    for target in targets:
        for dname, dirs, files in os.walk(target):
            for fname in files:
                fpath = os.path.join(dname, fname)
                replace_strings_in_file(fpath, old, replacement)
    targets = [
        "./pyproject.toml",
        "./docker-compose.yml",
        "./Dockerfile",
        "./scripts/stop.py",
        "./scripts/watch.py",
        "./scripts/make_migration_script.py",
        "./DEVELOP.md",
        "./README.md",
    ]
    for target in targets:
        replace_strings_in_file(target, old, replacement)
    os.rename(old, replacement)
