import argparse
import yaml
import torch
import sys
import importlib

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning_fabric as lf
from lightning.pytorch.callbacks import LearningRateMonitor

from engine import FeatureExtractor


## Parse arguments
parser = argparse.ArgumentParser(description = "Speaker verification with sequential module")

parser.add_argument('--config',         type=str,   default='./configs/mnist.yaml',   help='Config YAML file')
parser.add_argument('--mode',         type=str,   default='train',   help='choose train/val/eval')

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)


print(args)
print(config)
print('Python Version:', sys.version)
print('PyTorch Version:', torch.__version__)
print('Number of GPUs:', torch.cuda.device_count())





def train():
    # sets seeds for numpy, torch and python.random.    
    lf.utilities.seed.seed_everything(seed = config['random_seed'])

    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'

    training_dataset = importlib.import_module('dataloader.' + config['dataloader']).__getattribute__("training_dataset")
    training_dataset = training_dataset()
    test_dataset = importlib.import_module('dataloader.' + config['dataloader']).__getattribute__("test_dataset")
    test_dataset = test_dataset()

    train_dataloader = DataLoader(
            dataset = training_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )

    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )


    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.' + config['model']).__getattribute__("MainModel")
    model =  model()

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")

    # only for step scheduler(cosine_warmup_step.py)
    STEPS_PER_EPOCH = len(train_dataloader) # same as iteration per epoch
    TOTAL_EPOCH = config['max_epoch']
    config['scheduler_config']['warmup'] = config['scheduler_config']['warmup'] * STEPS_PER_EPOCH
    TOTAL_ITERATION = TOTAL_EPOCH * STEPS_PER_EPOCH

    scheduler = scheduler(optimizer, max_iters = TOTAL_ITERATION, **config['scheduler_config'])


    # ⚡⚡  3. Set 'engine' for training/validation and 'Trainer' 
    feature_extractor = FeatureExtractor(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler) 

    # ⚡⚡ 4. Init ModelCheckpoint callback, monitoring "val_ACC"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_ACC",
        mode="max",
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}-{val_ACC:02.2f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ⚡⚡ 5. LightningModule
    trainer = pl.Trainer(
        deterministic=True, # Might make your system slower, but ensures reproducibility.
        default_root_dir = config['default_root_dir'], #
        devices = config['devices'], #
        val_check_interval = 1.0, # Check val every n train epochs.
        max_epochs = config['max_epoch'], #
        sync_batchnorm = True, # ⚡⚡
        callbacks = [checkpoint_callback, lr_monitor], #
        accelerator = config['accelerator'], #
        num_sanity_val_steps = config['num_sanity_val_steps'], # Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check. 
        gradient_clip_val=1.0, # ⚡⚡
        profiler = config['profiler'], #
    )

    # ⚡⚡ 6. Resume training

    if config['resume_checkpoint']  is not None:
        trainer.fit(feature_extractor, train_dataloader, test_dataloader, ckpt_path=config['resume_checkpoint'])
        print(config['resume_checkpoint'] + "are loaded")
    else:
        trainer.fit(feature_extractor, train_dataloader, test_dataloader)
        print("no pre-trained weight are loaded")


def test():
    print("test")

    # sets seeds for numpy, torch and python.random.    
    lf.utilities.seed.seed_everything(seed = config['random_seed'])

    # ⚡⚡ 1. Set 'Dataset', 'DataLoader' 
    test_dataset = importlib.import_module('dataloader.' + config['dataloader']).__getattribute__("test_dataset")
    test_dataset = test_dataset()

    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )

    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.' + config['model']).__getattribute__("MainModel")
    model =  model()

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config['scheduler_config'])


    # ⚡⚡  3. Load model
    feature_extractor = FeatureExtractor.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'])

    # ⚡⚡ 4. LightningModule
    trainer = pl.Trainer(accelerator=config['accelerator'], devices = config['devices'])

    trainer.test(feature_extractor, dataloaders=test_dataloader)

    
if __name__ == "__main__":

    if args.mode == "train":
        train()

    elif args.mode == "test":
        test()

    # sets seeds for numpy, torch and python.random.
    