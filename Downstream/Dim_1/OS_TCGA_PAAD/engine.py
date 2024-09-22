import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from utils.logger import get_logger
from utils.pyt_utils import parse_devices, all_reduce_tensor, extant_file
from lifelines.utils import concordance_index

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


logger = get_logger()


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath


        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
            print("WORLD_SIZE is %d" % (int(os.environ['WORLD_SIZE'])))
        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))]

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        # p.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        # p.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    def data_parallel(self, model):
        if self.distributed:
            model = DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model)
        return model

    def get_train_loader(self, train_dataset, collate_fn=None, drop_last=False):
        train_sampler = None
        is_shuffle = True
        batch_size = self.args.batch_size

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            batch_size = self.args.batch_size // self.world_size
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=self.args.num_workers,
                                       drop_last=drop_last,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=train_sampler,
                                       collate_fn=collate_fn)

        return train_loader, train_sampler

    def get_test_loader(self, test_dataset, batch_size):
        test_sampler = None
        is_shuffle = False
        batch_size = batch_size

        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset)
            batch_size = self.args.batch_size // self.world_size

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       num_workers=self.args.num_workers,
                                       drop_last=False,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=test_sampler)

        return test_loader, test_sampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return torch.mean(tensor)

    # def train_one_epoch(self, model, train_loader, criterion, optimizer, epoch):
    #     model.train()
    #     total_loss = 0.0
    #     for batch in train_loader:
    #         inputs = batch['input_ids'].to(self.devices[0])
    #         attention_mask = batch['attention_mask'].to(self.devices[0])
    #         censor = batch['censor'].to(self.devices[0])
    #         survival_time = batch['survival_time'].to(self.devices[0])
    #
    #         optimizer.zero_grad()
    #         outputs = model(inputs, attention_mask=attention_mask)
    #         loss = criterion(outputs, survival_time, censor)
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #     avg_loss = total_loss / len(train_loader)
    #     logger.info(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
    #     return avg_loss
    #
    # def validate(self, model, val_loader, criterion):
    #     model.eval()
    #     total_loss = 0.0
    #     all_survival_times = []
    #     all_censor = []
    #     all_outputs = []
    #
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             inputs = batch['input_ids'].to(self.devices[0])
    #             attention_mask = batch['attention_mask'].to(self.devices[0])
    #             censor = batch['censor'].to(self.devices[0])
    #             survival_time = batch['survival_time'].to(self.devices[0])
    #
    #             outputs = model(inputs, attention_mask=attention_mask)
    #             loss = criterion(outputs, survival_time, censor)
    #
    #             total_loss += loss.item()
    #             all_survival_times.extend(survival_time.cpu().numpy())
    #             all_censor.extend(censor.cpu().numpy())
    #             all_outputs.extend(outputs.cpu().numpy())
    #
    #     avg_loss = total_loss / len(val_loader)
    #     c_index = concordance_index(all_survival_times, all_outputs, all_censor)
    #     logger.info(f"Validation Loss: {avg_loss:.4f}, C-index: {c_index:.4f}")
    #     return avg_loss, c_index

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "An exception occurred during Engine initialization, "
                "give up running process")
            return False