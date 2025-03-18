# Long short-term memory(LSTM) model algorithm

# Requirements

- Python >= 3.8
- Docker
- docker-compose
- [Poetry](https://python-poetry.org/)
- [Python TOML library](https://github.com/uiri/toml)
- [poethepoet](https://github.com/nat-n/poethepoet)

# Usage

## Initializing dependencies

```bash
poetry install --no-root
```

## Initializing `exodus_common`

```bash
git submodule update --init exodus_common
```

## Testing

Simply navigate to the desired algorithm repository and run the following command:
```bash
poe test
```

This command will take care of both training and prediction tasks for the respective algorithm.

### Dataset Structure
When testing, make sure your dataset directory, `tests/datasets/{SOME DATASET}`, contains the following files:

- `train.csv`: The input training data.
- `meta.json`: The configuration values for training.
- `prediction.csv`: The input predicting data.
Please refer to the provided dataset directories, such as AirPassengers, predict_unexpected_date, Shanghai_Car_License_Plate_Auction_Price, etc., for concrete examples of the dataset structure.

## Updating `exodus_common`


### `exodus_common`

```bash
poe update_exodus_common
```

## Managing Dependency

To add a dependency:
```bash
poetry add YOUR-PACKAGE
```

To update a dependency:
```bash
poetry add YOUR-PACKAGE@YOUR-VERSION       # Or `@latest`.
```

To remove a dependency:
```bash
poetry remove YOUR-PACKAGE
```

## TODO

- Debug
    - pdb
    - vscode attach to docker session
