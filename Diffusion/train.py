import os
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import argparse
import itertools
import json
import torchvision.transforms.functional as TF
from time import perf_counter as pc
from tqdm.auto import tqdm

from dataset import MyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Num workers."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0e-04,        #  khanh: taken from Zero123 (https://github.com/cvlab-columbia/zero123/blob/main/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml)
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help=("Path to train-test split file"),
    )
    parser.add_argument(
        "--cam", action="store_true", help="Whether to use cam_projection"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.instance_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    return args


class TrainerDDP:
    def __init__(
            self, 
            gpu_id, 
            pipe, 
            train_loader, 
            test_loader,
            train_set,
            test_set,
            train_sampler, 
            args
        ):
        self.args = args
        self.verbose = (gpu_id == 0)        
        self.gpu_id = gpu_id
        self.pipe = pipe
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_set = train_set
        self.test_set = test_set
        self.train_sampler = train_sampler
        self._print(f'[INFO] batch_size: {args.train_batch_size}, lr: {args.learning_rate}, accum: {args.gradient_accumulation_steps}')

        if self.args.resume_from_checkpoint:
            self.load_from_checkpoint(args.checkpoint)
        else:
            self._print(f'[INFO] Fine-tuning from huggingface')
            self.best_test_loss = float('inf')
            self.start_epoch = 0
            self.log_dict = {}
        
        self.pipe.to(self.gpu_id)

        ### Optimizer
        if self.args.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        # params_to_optimize = itertools.chain(
        #     self.pipe.unet.parameters(),
        # )
        params_to_optimize = self.pipe.get_params_to_optimize()
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        self._print(f"Train: {len(self.train_loader)}, Test: {len(self.test_loader)}")

    def run(self):
        num_steps_per_ep = math.ceil(len(self.train_loader) / self.args.gradient_accumulation_steps)
        total_steps = num_steps_per_ep * self.args.num_train_epochs
        starts_from = num_steps_per_ep * self.start_epoch
        # Only show the progress bar once on each machine.
        self.progress_bar = tqdm(range(starts_from, total_steps), disable=not (self.gpu_id == 0))
        self.progress_bar.set_description("Steps")        
        
        for epoch in range(self.start_epoch, self.args.num_train_epochs):
            self._print(f'[{epoch}/{self.args.num_train_epochs}]', end=' ')
            start_ep = pc()
            self.train_sampler.set_epoch(epoch)
            train_loss = self.train()
            # test_loss = self.test()
            test_loss = 0
            self.log_image(epoch)
            end_ep = pc() - start_ep        
            self._print(f'train loss {np.round(train_loss, 4)}, test loss {np.round(test_loss, 4)}, took {end_ep} seconds')
            self.save_checkpoint(train_loss, test_loss, epoch)
            
    def train(self):
        train_loss = 0
        self.pipe.train_modules()

        for step, batch in enumerate(self.train_loader, start=1):
            tgt = batch["tgt"]
            masked = batch["masked"]
            src = batch["src"]
            depth = batch["depth"]
            cam = batch["cam"]

            if self.args.cam:
                loss = self.pipe.train_step(tgt, src, masked, depth, cam)
            else:
                loss = self.pipe.train_step(tgt, src, masked, depth)
            train_loss = train_loss + loss.item() / len(self.train_set)
            loss.backward()

            if (step % self.args.gradient_accumulation_steps == 0) or (step == len(self.train_loader)): 
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
        
        return train_loss

    # def test(self):
    #     test_loss = 0
    #     self.pipe.unet.eval()
    #     self.pipe.clip_camera_projection.eval()
        
    #     with torch.no_grad():                
    #         for batch in self.test_loader:
    #             # tgt = batch["tgt"]
    #             # masked = batch["masked"]
    #             src = batch["src"]
    #             # depth = batch["depth"]
    #             cam = batch["cam"]

    #             if self.args.cam:
    #                 loss = self.pipe.train_step(tgt, src, masked, depth, cam)
    #             else:
    #                 loss = self.pipe.train_step(tgt, src, masked, depth)
    #             test_loss = test_loss + loss.item() / len(self.test_set)
                
    #     return test_loss
    
    def log_image(self, epoch):
        if self.gpu_id == 0:
            save_path = Path(self.args.output_dir) / "images" / f"ep-{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            for idx, batch in enumerate(self.test_loader):
                if idx == 10:
                    break
                masked = batch["masked"]
                src = batch["src"]
                tgt = batch["tgt"]
                depth = batch["depth"]
                cam = batch["cam"]
                cam_scale = (cam[0][0] / cam[0][1]).item()
                cam_scale = round(cam_scale, 4)
                
                if self.args.cam:
                    out = self.pipe(src, masked, depth, cam, use_progress_bar=False)['images'][0]
                else:
                    out = self.pipe(src, masked, depth, use_progress_bar=False)['images'][0]

                TF.to_pil_image(src.squeeze(0)).save(save_path / f"{idx}-src.png")
                TF.to_pil_image(tgt.squeeze(0)).save(save_path / f"{idx}-tgt.png")
                out.save(save_path / f"{idx}-output-{cam_scale}.png")
            
    def save_checkpoint(self, train_loss, test_loss, epoch):
        if self.gpu_id == 0:
            save_path = Path(self.args.output_dir) / "models"
            save_path.mkdir(parents=True, exist_ok=True)
            
            # save best
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                print(f'--> Lowest test loss = {np.round(test_loss, 4)} at epoch {epoch}')
                torch.save(self.pipe.unet.state_dict(), save_path / "unet_best.pt")

            # save last
            torch.save(self.pipe.unet.state_dict(), save_path / "unet_last.pt")

            self.log_dict[epoch] = {
                'Train loss': train_loss,
                'Test loss': test_loss
            }
            save_logging(self.log_dict, save_path / "logging.csv")
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


def get_loaders(args, train_set, test_set):

    train_sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(
        train_set, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True    
    )
    test_dataloader = DataLoader(
        test_set, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        sampler=DistributedSampler(test_set),
        pin_memory=True
    )
    return train_dataloader, test_dataloader, train_sampler


def save_logging(log_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(log_dict, f, sort_keys=True, indent=4)


def read_logging(logging):
    with open(logging, 'r') as f:
        data = json.load(f)
    
    best_test_loss = float('inf')
    new_data = {}

    for epoch in data.keys():
        new_data[int(epoch)] = data[epoch]

        test_loss = data[epoch]['Test loss']
        if test_loss < best_test_loss:
            best_test_loss = test_loss
    
    epoch = int(epoch) + 1
    
    return best_test_loss, epoch, new_data


# Each process control a single gpu
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main_ddp(rank, world_size, args):
    ddp_setup(rank, world_size)
    print(f'--> {rank=} {world_size=}')
    
    train_dataset = MyDataset(
        root=args.root,
        size=args.resolution,
        train=True
    )
    test_dataset = MyDataset(
        root=args.root,
        size=args.resolution,
        train=False
    )
    train_loader, test_loader, train_sampler = get_loaders(args, train_dataset, test_dataset)
    
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True    
    )

    if args.cam:
        from pipeline_camera import build_model
    else:
        from pipeline import build_model
    pipe = build_model(rank)
    trainer = TrainerDDP(
        gpu_id=rank,
        pipe=pipe,
        train_set=train_dataset,
        test_set=None,
        train_loader=train_loader,
        test_loader=train_loader,
        train_sampler=train_sampler,
        args=args
    )
    trainer.run()

    destroy_process_group()  # clean up

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_ddp,
        args=(world_size, args),
        nprocs=world_size
    )
    

    
    # ### debug
    # train_dataset = MyDataset(
    #     root=args.root,
    #     size=args.resolution,
    #     train=True
    # )
    # for d in train_dataset:
    #     pass