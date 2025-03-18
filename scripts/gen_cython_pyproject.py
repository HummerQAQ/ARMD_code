from collections import defaultdict

import toml
import tomli


def main(config_name: str) -> None:
    with open("pyproject.toml", "rb") as f:
        config = tomli.load(f)  # tomli is easier for parsing stuff

    body = defaultdict(dict)

    # Don't need poethepoet for cythonized build
    body["tool"]["poetry"] = config["tool"]["poetry"]
    # Don't need dev dependencies
    del body["tool"]["poetry"]["dev-dependencies"]

    body["tool"]["poetry"]["build"] = "build.py"
    body["build-system"] = config["build-system"]
    # Add cython dependency here
    body["build-system"]["requires"].append("cython")

    # Too bad there's no dump method in tomli
    with open(config_name, "w") as f:
        toml.dump(body, f)


if __name__ == "__main__":
    config_name = "cythonize-build-pyproject.toml"
    main(config_name)
