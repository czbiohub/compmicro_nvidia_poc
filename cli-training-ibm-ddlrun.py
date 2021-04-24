"""
DDL run training on ibm following tutorial
https://www.ibm.com/docs/en/wmlce/1.7.0?topic=ddl-tutorial-pytorch
$CONDA_PREFIX/lib/python$PY_VER/site-packages/torch/examples/ddl_examples/mnist/mnist.py

and data from dynamorph vq-vae model

"""
from dl_training.dynamorph.vqvae.ddlrun_train import main_worker
import argparse
import logging
from datetime import datetime
import sys, os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pyddl

"""
Dynamorph -- microglia -- VQ-VAE training
requirements:
- pytorch
"""

# logging
# One of 0, 10, 20, 30, 40, 50 for NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL respectively


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--model_output_dir',
        type=str,
        required=False,
        help="path to the directory to write trained models",
    )
    parser.add_argument(
        '-p', '--project_dir',
        type=str,
        required=False,
        help="path to the project folder containing the JUNE/raw subfolder",
    )
    parser.add_argument(
        '-c', '--channels',
        type=lambda s: [int(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of integers like '1,2,3' corresponding to channel indicies",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # set world_size retrieved from MPI
    if os.getenv('OMPI_COMM_WORLD_SIZE'):
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))

    start = datetime.now()

    # start logs
    logging.basicConfig(
        level=logging.DEBUG,
        # format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.model_output_dir}/{start.strftime('%Y_%m_%d_%H_%M')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # logging.basicConfig(filename=f"{args.model_output_dir}/{start.strftime('%Y_%m_%d_%H_%M')}")
    log = logging.getLogger(__name__)
    log.setLevel(20)
    log.info(f"================ BEGIN vq-vae training ============== ")
    log.info(f"================== {start.strftime('%Y_%m_%d_%H_%M')} ================= ")

    main_worker(args)

    stop = datetime.now()
    log.info(f"================ END vq-vae training ============== ")
    log.info(f"================== {stop.strftime('%Y_%m_%d_%H_%M')} ================= ")
    log.info(f"time elapsed = {(stop-start).days}-days_{(stop-start).seconds//60}-minutes_{(stop-start).seconds%60}-seconds")
