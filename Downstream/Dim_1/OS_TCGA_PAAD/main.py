import argparse
import os, sys
import numpy as np
from dataloader import TCGA_PAAD_Dataset
import os.path as osp
from Downstream.Dim_1.OS_TCGA_PAAD.model.Unimodel import Unified_Model
import timeit, time
from utils.ParaFlop import print_model_parm_nums
from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
from torch.cuda.amp import GradScaler, autocast
import shutil
import torch
from tqdm import tqdm
start = timeit.default_timer()
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch.backends.cudnn as cudnn
from utils.pyt_utils import CIndex

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Downstream TCGA-PAAD tasks")

    parser.add_argument("--data_path", type=str, default=r"D:\Analysis TCGA PAAD\Generated Files\data")
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/tmp/')

    parser.add_argument("--reload_from_pretrained", type=str2bool, default=True)
    parser.add_argument("--pretrained_path", type=str, default=r"C:\Users\Rafay\OneDrive\Desktop\Work\Moffitt Internship\MedCoss\Pretrained Model\checkpoint-299.pth")

    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--val_only", type=int, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    # others
    parser.add_argument("--gpu", type=str, default='None')

    parser.add_argument("--arch", type=str, default='unified_vit')
    parser.add_argument("--model_name", type=str, default='model')

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    if i_iter < 0:
        lr = 1e-2 * lr + i_iter * (lr - 1e-2 * lr) / 10.
    else:
        lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_alpha(i_iter, num_stemps):
    alpha_begin = 1
    alpha_end = 0.01
    decay = (alpha_begin - alpha_end) / num_stemps
    alpha = alpha_begin - decay * i_iter
    return alpha


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)
    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1
    dice = 2 * num / den
    return dice.mean()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        cudnn.benchmark = True

        # Model setup
        model = Unified_Model(num_classes=args.num_classes, pre_trained=args.reload_from_pretrained, pre_trained_weight=args.pretrained_path, model_name=args.model_name)
        print_model_parm_nums(model)

        model.train()

        # Concordance Index for train and dev sets
        train_ci, val_ci = -1, -1

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        # Optimizer: Only train the `cls_head`
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.cls_head.parameters()), lr=args.learning_rate, weight_decay=0.0001)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        # Load checkpoint if exists
        to_restore = {"epoch": 0}
        restart_from_checkpoint(os.path.join(args.snapshot_dir, "checkpoint.pth"), run_variables=to_restore, model=model, optimizer=optimizer)
        start_epoch = to_restore["epoch"]

        # Load data
        trainloader, train_sampler = engine.get_train_loader(TCGA_PAAD_Dataset(args.data_path, split="train_fold_1", max_words=args.input_size), drop_last=False)
        valloader, val_sampler = engine.get_test_loader(TCGA_PAAD_Dataset(args.data_path, split="dev_fold_1", max_words=args.input_size), batch_size=1)
        testloader, test_sampler = engine.get_test_loader(TCGA_PAAD_Dataset(args.data_path, split="test_set", max_words=args.input_size), batch_size=1)

        print("train dataset len: {}, val dataset len: {}".format(len(trainloader), len(valloader)))
        
        best_ci = -1
        for epoch in range(start_epoch, args.num_epochs):
            if args.val_only == 1:
                break

            time_t1 = time.time()
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            model.train()

            # Training loop (only loss computation here)
            for iter, (input_ids, survival_time, censor, att_masks) in tqdm(enumerate(trainloader)):
                # print(f"Input: {input_ids},\nSurv: {survival_time}, \nCensor: {censor}, \nAtt: {att_masks}")
                # input()
                input_ids = input_ids.cuda(non_blocking=True)
                survival_time = survival_time.cuda(non_blocking=True)
                censor = censor.cuda(non_blocking=True)
                att_masks = att_masks.cuda(non_blocking=True)

                data = {"data": input_ids, "survival_time": survival_time, "censor": censor,
                        "mask_attention": att_masks}
                optimizer.zero_grad()

                # Compute Cox Loss
                cox_loss = model(data)
                cox_loss.backward()
                optimizer.step()

                epoch_loss.append(float(cox_loss))

            epoch_loss = np.mean(epoch_loss)
            print(f"Epoch {epoch}: Cox Loss = {epoch_loss:.4f}")

            # After the epoch, compute CI for the training set
            model.eval()  # Set model to evaluation mode during CI computation
            pre_hazards_train = []
            survival_times_train = []
            censor_vals_train = []

            with torch.no_grad():
                for iter, (input_ids, survival_time, censor, att_masks) in tqdm(enumerate(trainloader)):
                    input_ids = input_ids.cuda(non_blocking=True)
                    survival_time = survival_time.cuda(non_blocking=True)
                    censor = censor.cuda(non_blocking=True)
                    att_masks = att_masks.cuda(non_blocking=True)

                    data = {"data": input_ids, "survival_time": survival_time, "censor": censor,
                            "mask_attention": att_masks}

                    # Get hazard scores for CI calculation
                    hazard_scores = model.inference(data)
                    pre_hazards_train.extend(hazard_scores.cpu().numpy())  # Handle the entire batch
                    survival_times_train.append(survival_time.cpu().numpy())
                    censor_vals_train.append(censor.cpu().numpy())

            pre_hazards_train = np.array(pre_hazards_train)
            survival_times_train = np.concatenate(survival_times_train)
            censor_vals_train = np.concatenate(censor_vals_train)
            train_ci = CIndex(pre_hazards_train, censor_vals_train, survival_times_train)
            print(f"Training CI: {train_ci:.4f}")

            # Validation with Concordance Index
            model.eval()
            pre_hazards = []
            survival_times = []
            censor_vals = []

            with torch.no_grad():
                for iter, (input_ids, survival_time, censor, att_masks) in tqdm(enumerate(valloader)):
                    # Move data to GPU
                    input_ids = input_ids.cuda(non_blocking=True)
                    survival_time = survival_time.cuda(non_blocking=True)
                    censor = censor.cuda(non_blocking=True)
                    att_masks = att_masks.cuda(non_blocking=True)

                    # Prepare the data dictionary
                    data = {
                        "data": input_ids,
                        "survival_time": survival_time,
                        "censor": censor,
                        "mask_attention": att_masks,
                    }

                    # Use the inference method to get hazard scores
                    hazard_scores = model.inference(data)

                    # Store the results, make sure to extract scalar values
                    pre_hazards.append(hazard_scores.item())  # Append scalar instead of array
                    survival_times.append(survival_time.cpu().numpy())
                    censor_vals.append(censor.cpu().numpy())

            # No need for np.concatenate since pre_hazards are scalars now
            pre_hazards = np.array(pre_hazards)
            survival_times = np.concatenate(survival_times)
            censor_vals = np.concatenate(censor_vals)

            # Compute Concordance Index
            val_ci = CIndex(pre_hazards, censor_vals, survival_times)
            print(f"Validation CI: {val_ci:.4f}")

            # Save the best model based on CI
            if val_ci > best_ci:
                best_ci = val_ci
                print(f"Best CI achieved: {val_ci:.4f}. Saving model...")
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                }
                torch.save(save_dict, os.path.join(args.snapshot_dir, 'checkpoint.pth'))

        # Test Phase: Evaluate using Concordance Index
        model.eval()
        pre_hazards = []
        survival_times = []
        censor_vals = []

        with torch.no_grad():
            for iter, (input_ids, survival_time, censor, att_masks) in tqdm(enumerate(testloader)):
                input_ids = input_ids.cuda(non_blocking=True)
                survival_time = survival_time.cuda(non_blocking=True)
                censor = censor.cuda(non_blocking=True)
                att_masks = att_masks.cuda(non_blocking=True)

                # Prepare the data dictionary
                data = {"data": input_ids, "survival_time": survival_time, "censor": censor,
                        "mask_attention": att_masks}

                # Use the inference method to get hazard scores
                hazard_scores = model.inference(data)

                # Store the results, make sure to extract scalar values
                pre_hazards.append(hazard_scores.item())  # Append scalar instead of array
                survival_times.append(survival_time.cpu().numpy())
                censor_vals.append(censor.cpu().numpy())

        # No need for np.concatenate since pre_hazards are scalars now
        pre_hazards = np.array(pre_hazards)
        survival_times = np.concatenate(survival_times)
        censor_vals = np.concatenate(censor_vals)

        # Compute Concordance Index for test set
        test_ci = CIndex(pre_hazards, censor_vals, survival_times)
        print(f"Test CI: {test_ci:.4f}")
        with open(os.path.join(args.snapshot_dir, "result.txt"), "w") as fp:
            fp.write(f"Training CI: {train_ci:.4f}\n")
            fp.write(f"Validation CI: {val_ci:.4f}")
            fp.write(f"Test CI: {test_ci:.4f}")


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    checkpoint = torch.load(ckp_path, map_location="cpu")
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


if __name__ == '__main__':
    main()
