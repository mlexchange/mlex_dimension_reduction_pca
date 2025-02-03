import argparse
import logging
import os
import time

import yaml

from dim_reduction import compute_pca
from parameters import IOParameters, PCAParameters
from utils.data_utils import load_data, save_results
from utils.tiled_utils import TiledDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Load parameters
    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Validate parameters
    io_parameters = IOParameters(**parameters["io_parameters"])
    model_parameters = PCAParameters(**parameters["model_parameters"])
    logger.info(f"Parameters loaded: {model_parameters}")

    # Load images from given data_uris
    if io_parameters.data_type == "file":
        data_uri = None
    else:
        data_uri = io_parameters.root_uri
    tiled_dataset = TiledDataset(
        data_uri,
        io_parameters.results_tiled_uri,
        read_tiled_key=io_parameters.data_tiled_api_key,
        write_tiled_key=io_parameters.results_tiled_api_key,
    )
    stacked_images = load_data(io_parameters, tiled_dataset, logger)

    start_time = time.time()

    # Run PCA
    if io_parameters.save_model_path is not None:
        os.makedirs(io_parameters.save_model_path, exist_ok=True)
        save_model_path = (
            f"{io_parameters.save_model_path}/{io_parameters.uid_save}.joblib"
        )
    else:
        save_model_path = None
    latent_vectors = compute_pca(
        stacked_images,
        n_components=model_parameters.n_components,
        load_model_path=io_parameters.load_model_path,
        save_model_path=save_model_path,
    )

    save_results(latent_vectors, io_parameters, tiled_dataset, parameters)

    logger.info("PCA done!")
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")
