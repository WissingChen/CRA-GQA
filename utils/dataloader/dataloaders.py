import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import VideoQADataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)



def build_dataloaders(cfgs, a2id, tokenizer, ddp=False):
    train_dataset = VideoQADataset(cfgs, "train", tokenizer, a2id)
    val_dataset = VideoQADataset(cfgs, "val", tokenizer, a2id)
    test_dataset = VideoQADataset(cfgs, "test", tokenizer, a2id) if cfgs["dataset"]["name"] != "star" else None

    if ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)

        # train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, cfgs["dataset"]["batch_size"], drop_last=True)
    
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False if ddp else True,
        drop_last=True,
        collate_fn=videoqa_collate_fn,
        batch_sampler=train_sampler, 
        pin_memory=True if ddp else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False,
        collate_fn=videoqa_collate_fn,
        batch_sampler=val_sampler, 
        pin_memory=True if ddp else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False,
        drop_last=False,
        collate_fn=videoqa_collate_fn,
        batch_sampler=test_sampler, 
        pin_memory=True if ddp else False
    ) if cfgs["dataset"]["name"] != "STAR" else None


    return train_loader, val_loader, test_loader