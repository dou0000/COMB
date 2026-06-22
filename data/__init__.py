

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np
import random


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataloader(opt):
    def _seed_worker(worker_id: int):
        # one seed per worker that is still reproducible across runs
        base_seed   = opt.seed            # ← the seed you passed in
        worker_seed = (base_seed + worker_id) % 2**32

        # numpy & builtin random
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # torch
        torch.manual_seed(worker_seed)


    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    
    g = torch.Generator().manual_seed(opt.seed)


    return torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
        worker_init_fn=_seed_worker,
        generator=g
    )
