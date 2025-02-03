from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PCAParameters(BaseModel):
    n_components: int = Field(description="number of components to keep")


class DataType(str, Enum):
    file = "file"
    tiled = "tiled"


class IOParameters(BaseModel):
    uid_retrieve: str = Field(description="uid to retrieve data from")
    data_type: DataType = Field(description="data type")
    root_uri: str = Field(description="root uri")
    data_uris: list[str] = Field(description="data uris")
    data_tiled_api_key: Optional[str] = Field(description="tiled api key", default=None)
    results_tiled_uri: str = Field(description="tiled uri to save data to")
    results_tiled_api_key: Optional[str] = Field(
        description="tiled api key", default=None
    )
    uid_save: str = Field(description="uid to save data to")
    results_dir: str = Field(description="results directory")
    load_model_path: Optional[str] = None
    save_model_path: Optional[str] = None
    project_name: Optional[str] = None
