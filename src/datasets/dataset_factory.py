import json
from typing import Any
from pydantic import BaseModel
from src.datasets.dataset import Dataset, DatasetAttribute


class DatasetsFactory(BaseModel):
    @staticmethod
    def build_dataset(dataset_name: str) -> Any:
        try:
            ds = Dataset.get_dataset_by_name(dataset_name)
            print(ds, dataset_name)
            if ds:
                raise AssertionError("Dataset name is not valid.")
            return ds
        except AssertionError as e:
            print(e)
