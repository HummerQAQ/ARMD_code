from exodusutils.migration.base_migration import BaseMigration, Collection
from pymongo import UpdateOne


class Migration(BaseMigration):
    def __init__(
        self, timestamp: int = 1663732251511, name: str = "schema_for_new_k8s"
    ) -> None:
        super().__init__(timestamp, name)

    def up(self, collection: Collection) -> None:
        bulkOps = []
        for doc in collection.find({"target_type": {"$exists": True}}):
            bulkOps.append(
                UpdateOne(
                    filter={"_id": doc["_id"]},
                    update={
                        "$set": {
                            "name": "lstm",
                            "target": {
                                "name": doc["target"],
                                "data_type": doc["target_type"],
                            },
                            "groupby": "standardgroupby",
                            "date_column_name": doc["date_column"],
                            "final_model_data": str(doc["train_data"]),
                            "scaler": str(doc["scaler"]),
                            "model": str(doc["model"]),
                        },
                        "$unset": {
                            "target_type": 1,
                            "group_by": 1,
                            "date_column": 1,
                            "derivation_window": 1,
                            "train_data": 1,
                        },
                    },
                )
            )
        if bulkOps:
            collection.bulk_write(bulkOps)

    def down(self, collection: Collection) -> None:
        # TODO implement this method
        pass
