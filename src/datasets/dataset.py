import json
import os
from typing import Optional

from pydantic import BaseModel
from dataclasses import dataclass


class DatasetAttribute(BaseModel):
    path:str 
    test:Optional[str]
    pipe:str
    type:str
    

class Dataset(BaseModel):
    name:Optional[str]
    dataset_attribute:Optional[DatasetAttribute]
    
    def get_datasets(self):
        return self.dataset_attribute.dict()
    
    def get_dataset_by_name(self, name):
        with open("src/datasets/data/datasets.json") as fp:
            dictObj = json.load(fp)
            self.name = name
            self.dataset_attribute = DatasetAttribute(**dictObj[name])
        return self
        
    def add_dataset(self):
        with open("src/datasets/data/datasets.json") as fp:
            dictObj = json.load(fp)
            dictObj[self.name] = self.dataset_attribute.dict()
        with open("src/datasets/data/datasets.json", "w") as fp:
            json.dump(dictObj, fp)