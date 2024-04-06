import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from tiled.client import from_uri

from utils import load_images_from_directory

""" Compute PCA
    Input: 1d data (N, M) or 2d data (N, H, W)
    Output: latent vectors of shape (N, 2) or (N, 3)
"""


def computePCA(data, n_components=2, standarize=False):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    # Check input and output dir are provided
    assert io_parameters[
        "output_dir"
    ], "Output dir (dir to save the computed latent vactors) not provided for training."

    # Validate model parameters:
    model_parameters = parameters["model_parameters"]
    print("model_parameters")
    print(model_parameters)

    # output directory
    output_dir = pathlib.Path(
        io_parameters["output_dir"] + "/" + io_parameters["uid_save"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images from given data_uris
    stacked_images = None

    uid_retrieve = io_parameters["uid_retrieve"]
    if uid_retrieve != "":
        # Get feature vectors from autoencoder
        auto_fv_dir = io_parameters["output_dir"] + "/" + uid_retrieve
        stacked_images = pd.read_parquet(f"{auto_fv_dir}/f_vectors.parquet").values

    else:
        data_uris = io_parameters["data_uris"]

        for uri in data_uris:
            if "Demoshapes.npz" in uri:  # example dataset
                images = np.load(uri)["arr_0"]

            else:
                # FM, file system or tiled
                if io_parameters["data_type"] == "file":
                    images = load_images_from_directory(
                        io_parameters["root_uri"] + "/" + uri
                    )
                else:  # tiled
                    tiled_client = from_uri(
                        io_parameters["root_uri"],
                        api_key=io_parameters["data_tiled_api_key"],
                    )
                    images = tiled_client[uri][:]
                    if len(images.shape) == 2:
                        images = images[np.newaxis, :, :]

            if stacked_images is None:
                stacked_images = images
            else:
                stacked_images = np.concatenate((stacked_images, images), axis=0)

    start_time = time.time()

    # Run PCA
    latent_vectors = computePCA(
        stacked_images, n_components=model_parameters["n_components"]
    )
    print("Latent vector shape: ", latent_vectors.shape)

    # Save latent vectors
    output_name = "latent_vectors.npy"
    save_path = str(output_dir) + "/" + output_name
    print(save_path)
    np.save(str(output_dir) + "/" + output_name, latent_vectors)

    print("PCA done, latent vector saved.")
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution time: {execution_time} seconds")
