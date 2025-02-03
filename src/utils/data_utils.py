import os
import pathlib

import numpy as np
import pandas as pd
from PIL import Image


def load_data(io_parameters, tiled_dataset, logger=None):
    """
    Load data from the given data_uris
    :param io_parameters: IOParameters object
    :param tiled_dataset: TiledDataset object
    :param logger: logger object
    :return: stacked_images: numpy array
    """

    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    uid_retrieve = io_parameters.uid_retrieve

    # Get feature vectors from autoencoder
    if uid_retrieve != "":
        write_dir = io_parameters.results_dir
        retrieve_path = f"{write_dir}/{uid_retrieve}/latent_vectors.parquet"
        stacked_images = pd.read_parquet(retrieve_path).values
        logger.info(f"Feature vectors loaded from {retrieve_path}")

    else:
        stacked_images = None
        if io_parameters.data_type == "file":
            for uri in io_parameters.data_uris:
                images = load_images_from_directory(
                    io_parameters.root_uri + "/" + uri,
                    logger,
                )
                if stacked_images is None:
                    stacked_images = images
                else:
                    stacked_images = np.concatenate((stacked_images, images), axis=0)
            logger.info(f"Images loaded from {io_parameters.root_uri}")
        else:  # tiled
            for uri in io_parameters.data_uris:
                images = tiled_dataset.load_data_from_tiled(io_parameters.data_uris[0])
                if len(images.shape) == 2:
                    images = images[np.newaxis,]
                if stacked_images is None:
                    stacked_images = images
                else:
                    stacked_images = np.concatenate((stacked_images, images), axis=0)
            logger.info(f"Images loaded from {io_parameters.root_uri}")
    logger.info(f"Data shape: {stacked_images.shape}")
    return stacked_images


def load_images_from_directory(directory_path, logger):
    """
    Load images from the given directory path
    :param directory_path: str
    :return: image_data: numpy array
    """
    image_data = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith((".png", ".tif", ".tiff", ".jpeg", ".jpg")):
            file_path = os.path.join(directory_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                image_data.append(img_array)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    image_data = np.array(image_data)
    return image_data


def save_results(latent_vectors, io_parameters, tiled_dataset, parameters):
    """
    Save latent vectors to the given directory
    :param latent_vectors: numpy array
    :param io_parameters: IOParameters object
    :param tiled_dataset: TiledDataset object
    :param parameters: dict
    :return: latent_vectors_path: str
    """
    write_dir = io_parameters.results_dir
    uid_save = io_parameters.uid_save

    # Output directory
    output_dir = pathlib.Path(f"{write_dir}/{uid_save}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latent vectors to directory
    latent_vectors_path = f"{output_dir}/latent_vectors.parquet"
    column_names = [f"feature_{i}" for i in range(latent_vectors.shape[1])]
    latent_vectors = pd.DataFrame(latent_vectors, columns=column_names)
    latent_vectors.to_parquet(latent_vectors_path)

    tiled_dataset.write_results(
        latent_vectors, io_parameters, latent_vectors_path, parameters
    )
    return latent_vectors_path
