import logging
import os
import torch
import time
import wandb

from models import get_model
from dataloaders import get_dataclass
from tqdm import tqdm

from utils.logging import config_logging
from utils.loss import get_loss
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler


# class TqdmToLogger(io.StringIO):
#     """
#         Output stream for TQDM which will output to logger module instead of
#         the StdOut.
#     """
#     logger = None
#     level = None
#     buf = ''
#     def __init__(self, logger, level=None):
#         super(TqdmToLogger, self).__init__()
#         self.logger = logger
#         self.level = level or logging.INFO
#     def write(self, buf):
#         self.buf = buf.strip('\r\n\t ')
#     def flush(self):
#         self.logger.log(self.level, self.buf)


class Trainer():
    def __init__(self, config, output_dir):
        self.config = config
        self.train_epochs = self.config['nepochs']
        self.warmup_epochs = self.config['warmup_epochs']
        self.warmup_lr = self.config['warmup_lr']
        self.lr = self.config['lr']
        # self.lr_restart = self.config['lr_restart']
        self.lr_min = self.config['lr_min']
        # self.global_train_step = 0

        self.save_interval = self.config['save_every']
        self.prev_save = -1

        self.output_dir = output_dir

        # Setup logging
        log_file = os.path.join(self.output_dir, 'info.log')
        config_logging(verbose=self.config['verbose'], log_file=log_file, append=self.config['resume'])
        logging.info(f"Initialized the logging for experiment: {self.config['name']}")

        # # Create a tqdm handler
        # logger = logging.getLogger()
        # self.tqdm_out = TqdmToLogger(logger,level=logging.INFO)

        # Check if the script is part of a sweep
        is_sweeping = os.getenv('WANDB_SWEEP_ID') is not None

        # Initialize a run if not part of a sweep
        if not is_sweeping:
            logging.info("Initialize the individual WandB run")
            wandb.init(project="NesT_HPML", name=self.config['name'])

        # Initialize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataClass = get_dataclass(config)

        logging.info(f"Model: {self.config['model']}")
        
        self.model = get_model(self.config['model'], pretrained=False)
        self.model.to(self.device)

        self.train_dl = self.dataClass.make_dataloader(train=True)
        self.test_dl = self.dataClass.make_dataloader(train=False)

        logging.info(f"Number of train samples in the {config['dataset']} dataset: {len(self.train_dl.dataset)}")
        logging.info(f"Number of test samples in the {config['dataset']} dataset: {len(self.test_dl.dataset)}")

        self.loss = get_loss(self.config['loss'])

        self.optimizer = get_optimizer(self.config['optimizer'])
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])

        self.warmup_scheduler = get_scheduler(self.config['warmup_scheduler'])
        self.warmup_scheduler = self.warmup_scheduler(self.optimizer, start_factor=self.warmup_lr/self.lr, end_factor=1.0, total_iters=self.warmup_epochs)

    
    def train(self):
        total_train_time = total_test_time = total_deviceload_time = total_dataload_time = total_overhead_time = 0.0
        
        # Warm-Up
        logging.info(f"Warm up the model for {self.warmup_epochs} epochs")
        for e in range(self.warmup_epochs):
            self.e = e
            wandb.log({"lr": self.warmup_scheduler.get_last_lr()[0]}, step=self.e)
            epoch_train_time, epoch_train_dataload_time, epoch_train_deviceload_time, epoch_train_overhead_time = self.per_epoch(key='train')
            epoch_test_time, epoch_test_dataload_time, epoch_test_deviceload_time, epoch_test_overhead_time = self.per_epoch(key='test')
            self.warmup_scheduler.step()
            total_train_time += epoch_train_time
            total_test_time += epoch_test_time
            total_dataload_time += epoch_train_dataload_time + epoch_test_dataload_time
            total_deviceload_time += epoch_train_deviceload_time + epoch_test_deviceload_time
            total_overhead_time += epoch_train_overhead_time + epoch_test_overhead_time

        logging.info(f"Warm up: Average training time per epoch: {total_train_time/self.warmup_epochs:.3f} s")
        logging.info(f"Warm up: Average test time per epoch: {total_test_time/self.warmup_epochs:.3f} s")
        logging.info(f"Warm up: Average data-loading time per epoch: {total_dataload_time/self.warmup_epochs:.3f} s")
        logging.info(f"Warm up: Average device-loading time per epoch: {total_deviceload_time/self.warmup_epochs:.3f} s")
        # logging.info(f"Warm up: Average overhead time per epoch: {total_overhead_time/self.warmup_epochs:.3f} s")

        logging.info(".................................................................")
        logging.info(f"Warm up completed. Now training for {self.train_epochs} epochs")
        logging.info(f"Initializing the training learning rate scheduler")

        self.train_scheduler = get_scheduler(self.config['train_scheduler'])
        self.train_scheduler = self.train_scheduler(self.optimizer, T_max=self.train_epochs, eta_min=self.lr_min)

        # Actual training
        for e in range(self.warmup_epochs, self.train_epochs):
            self.e = e
            wandb.log({"lr": self.train_scheduler.get_last_lr()[0]}, step=self.e)
            epoch_train_time, epoch_train_dataload_time, epoch_train_deviceload_time, epoch_train_overhead_time = self.per_epoch(key='train')
            epoch_test_time, epoch_test_dataload_time, epoch_test_deviceload_time, epoch_test_overhead_time = self.per_epoch(key='test')
            self.train_scheduler.step()
            if e - self.prev_save == self.save_interval:
                self.prev_save = e
                torch.save({
                    'epoch': e,
                    'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                }, os.path.join(self.output_dir, '{}.tar'.format(f'checkpoint_{e}')))
            
            total_train_time += epoch_train_time
            total_test_time += epoch_test_time
            total_dataload_time += epoch_train_dataload_time + epoch_test_dataload_time
            total_deviceload_time += epoch_train_deviceload_time + epoch_test_deviceload_time
            total_overhead_time += epoch_train_overhead_time + epoch_test_overhead_time

        logging.info(f"Full Training: Average training time per epoch: {total_train_time/self.train_epochs:.3f} s")
        logging.info(f"Full Training: Average test time per epoch: {total_test_time/self.train_epochs:.3f} s")
        logging.info(f"Full Training: Average data-loading time per epoch: {total_dataload_time/self.train_epochs:.3f} s")
        logging.info(f"Full Training: Average device-loading time per epoch: {total_deviceload_time/self.train_epochs:.3f} s")
        # logging.info(f"Full Training: Average overhead time per epoch: {total_overhead_time/self.train_epochs:.3f} s")

        logging.info("Upload the experiment information to WandB")
        artifact = wandb.Artifact(name=self.config['name'], type="info")
        artifact.add_dir(self.output_dir)
        wandb.log_artifact(artifact)

    def per_epoch(self, key):
        epoch_mode_time = epoch_deviceload_time = epoch_dataload_time = epoch_overhead_time = 0.0
        dataloader = {'train': self.train_dl, 'test': self.test_dl}

        self.iterator = iter(dataloader[key])
        total_iterations = len(dataloader[key])
        current_iteration = 0
        epoch_mode_accuracy = 0
        epoch_mode_loss = 0
        
        self.model.train() if key == 'train' else self.model.eval()
        inference_mode = key != 'train'        

        with torch.inference_mode(mode=inference_mode):
            with tqdm(len(dataloader[key])) as tepoch:
                tepoch.set_description(f"Epoch {self.e}: {key}")
                while(current_iteration < total_iterations):
                    current_iteration += 1
                    
                    if self.device == 'cuda':   torch.cuda.synchronize()
                    dataload_start = time.perf_counter()
                    data, target = next(self.iterator)
                    if self.device == 'cuda':   torch.cuda.synchronize()
                    dataload_end = time.perf_counter()

                    input = data.to(self.device)
                    target = target.to(self.device)
                    if self.device == 'cuda':   torch.cuda.synchronize()
                    mode_start = time.perf_counter()

                    if key == 'train': self.optimizer.zero_grad()
                    
                    output = self.model(input)

                    minibatch_loss = self.loss(output, target)
                    if key == 'train': minibatch_loss.backward()

                    if key == 'train': self.optimizer.step()

                    if self.device == 'cuda':   torch.cuda.synchronize()
                    mode_end = time.perf_counter()

                    minibatch_accuracy = torch.sum(torch.argmax(output, dim=1) == target) / target.shape[0]
                    
                    epoch_mode_accuracy += 100. * minibatch_accuracy.item() * target.shape[0]
                    epoch_mode_loss += minibatch_loss.item() * target.shape[0]
                    
                    tepoch.update(1)
                    tepoch.set_postfix(mb_loss=minibatch_loss.item(), 
                                        mb_accuracy=100. * minibatch_accuracy.item())
                    
                    if self.device == 'cuda':   torch.cuda.synchronize()
                    overhead_end = time.perf_counter()
                    # if key == 'train': self.global_train_step += 1

                    epoch_dataload_time += (dataload_end - dataload_start)
                    epoch_mode_time += (mode_end - mode_start)
                    epoch_deviceload_time += (mode_start - dataload_end)
                    epoch_overhead_time += (overhead_end - mode_end)

            epoch_mode_accuracy /= len(dataloader[key].dataset)
            epoch_mode_loss /= len(dataloader[key].dataset)

        wandb.log({f"{key}-epoch-accuracy": epoch_mode_accuracy}, step=self.e)
        wandb.log({f"{key}-epoch-loss": epoch_mode_loss}, step=self.e)

        return epoch_mode_time, epoch_dataload_time, epoch_deviceload_time, epoch_overhead_time
