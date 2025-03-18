import argparse
import datetime
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Create a migration script")
    parser.add_argument("name", help="Name of the migration script")
    args = parser.parse_args()

    templates_path = "scripts/migration_templates"
    env = Environment(
        loader=FileSystemLoader(templates_path),
        autoescape=select_autoescape(),
        trim_blocks=True,
    )

    script_template = env.get_template("migration_template.py.j2")
    timestamp = int(datetime.datetime.now().timestamp() * 1000)
    name = args.name
    destination = "lstm/migrations"
    script_prefix = "Migration_"
    script_name = f"{script_prefix}{timestamp}_{name}"
    script_file_path = f"{destination}/{script_name}.py"
    with open(script_file_path, "w") as f:
        f.write(script_template.render(timestamp=timestamp, name=name))
    print(f"Created: {script_file_path}")

    migrations_template = env.get_template("all_migrations.py.j2")
    scripts = [
        f[:-3]
        for f in os.listdir(destination)
        if f[: len(script_prefix)] == script_prefix and f[-3:] == ".py"
    ]
    migrations_file_path = f"{destination}/all_migrations.py"
    with open(migrations_file_path, "w") as f:
        f.write(migrations_template.render(scripts=scripts))
    print(f"Updated: {migrations_file_path}")
