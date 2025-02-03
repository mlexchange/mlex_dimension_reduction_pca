import numpy as np
from tiled.client import from_uri
from tiled.client.array import ArrayClient
from tiled.client.container import Container
from tiled.structures.data_source import Asset, DataSource
from tiled.structures.table import TableStructure


class TiledDataset:
    def __init__(
        self, read_tiled_uri, write_tiled_uri, read_tiled_key=None, write_tiled_key=None
    ):
        if read_tiled_uri:
            self.read_client = from_uri(read_tiled_uri, api_key=read_tiled_key)
        self.write_client = from_uri(write_tiled_uri, api_key=write_tiled_key)

    def load_data_from_tiled(self, uri):
        data_client = self.read_client[uri]
        if isinstance(data_client, Container):
            data = []
            sub_uris = list(data_client)
            for sub_uri in sub_uris:
                if isinstance(data_client[sub_uri], ArrayClient):
                    data.append(self.read_client[uri][sub_uri][:])
            return np.stack(data, axis=0)
        elif isinstance(data_client, ArrayClient):
            return self.read_client[uri][:]
        else:
            raise ValueError(f"Data structure not supported: {type(data_client)}")

    def write_results(
        self, latent_vectors, io_parameters, latent_vectors_path, metadata=None
    ):
        uid_save = io_parameters.uid_save

        # Save latent vectors to Tiled
        structure = TableStructure.from_pandas(latent_vectors)

        # Remove API keys from metadata
        if metadata:
            metadata["io_parameters"].pop("data_tiled_api_key", None)
            metadata["io_parameters"].pop("results_tiled_api_key", None)

        frame = self.write_client.new(
            structure_family="table",
            data_sources=[
                DataSource(
                    structure_family="table",
                    structure=structure,
                    mimetype="application/x-parquet",
                    assets=[
                        Asset(
                            data_uri=f"file://{latent_vectors_path}",
                            is_directory=False,
                            parameter="data_uris",
                            num=1,
                        )
                    ],
                )
            ],
            metadata=metadata,
            key=uid_save,
        )

        frame.write(latent_vectors)
        pass
