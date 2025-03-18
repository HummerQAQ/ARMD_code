# Migration

## Why?

When you're updating a model algorithm, it can be the case that you want to change the field names or content
in the `ModelInfo` class. This is when you will need a migration script to make sure things are consistent
before and after your changes are made.

## Create a migration script

```bash
python scripts/make_migration_script.py {NAME}
```

Once you create a migration script, the `{MODEL_ALGORITHM}/migrations` directory will contain the new script you
just generated, and the `{MODEL_ALGORITHM}/migrations/all_migrations.py` file will be updated.

As the model algorithm has hot reloading capability, it will load the newly created script into the container.

The rule of thumb for writing migration scripts is that whatever you're doing in the `up` step, you should either
do the opposite in the `down` step, or leave `down` as a no-op if the `up` step is meant to be something irreversible.

## Run migration actions

Make sure you have your model algorithm service up and running before doing the following steps!

### Migrate up

Send a cURL request to the `/migrate` endpoint, with JSON data `'{"action": "up"}'`.

```bash
$ curl -x POST "http://{YOUR HOST}:{YOUR PORT}/migrate" -H "Content-Type: application/json" -d '{"action": "up"}'
```

This will run the `up` step in all of the migration scripts in `{MODEL_ALGORITHM}/migrations/all_migrations.py`
that are not yet recorded in the `migration` collection in your MongoDB.

### Migrate down

Send a cURL request to the `/migrate` endpoint, with JSON data `'{"action": "down"}'`.

```bash
$ curl -x POST "http://{YOUR HOST}:{YOUR PORT}/migrate" -H "Content-Type: application/json" -d '{"action": "down"}'
```

This will run the `down` step defined in the most recent migration script that is recorded in the `migration`
collection in your MongoDB.
