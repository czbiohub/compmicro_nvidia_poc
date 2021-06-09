"""
Based on https://github.com/pytorch/examples/blob/master/imagenet/main.py

Distributed Data Parallel training on Imagenet
Use Distributed Deep Learning - ddl backend

Modifications:
*****************************************************************

Licensed Materials - Property of IBM

(C) Copyright IBM Corp. 2018. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

*****************************************************************

"""
from .vq_vae import VQ_VAE
import pdb

import os
import torch as t
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import glob
import math

import horovod.torch as hvd
from distutils.version import LooseVersion


from dl_training.dynamorph.vqvae.utils import DatasetFolderWithPaths, npy_loader
import numpy as np
import logging

log = logging.getLogger(__name__)


def train(model,
          train_loader,
          optimizer,
          relation_mat=None,
          mask_loader=None,
          args_=None,
          ):

    # model.zero_grad()
    model = model.float()
    optimizer.zero_grad()
    mean_loss = {'recon_loss': [],
                 'commitment_loss': [],
                 'time_matching_loss': [],
                 'total_loss': [],
                 'perplexity': []}
    # =====================================================

    for i, (patch, mask) in enumerate(zip(train_loader, mask_loader)):
        # patch and mask are tuples of shape (image_batch, class, filename_batch)

        patch_data = patch[0]
        mask_data = mask[0]
        patch_fn = patch[2]
        mask_fn = mask[2]
        if args_.cuda:
            patch_data, mask_data = patch_data.float().cuda(), mask_data.float().cuda()

        # set number of channels to use
        use_channels = list(range(1))
        n_channels = 1
        x_size, y_size = 128, 128

        # =================================
        # The loader's batch size is batch_size * batches_per_allreduce
        # Split data into sub-batches of size batch_size in case batches_per_allreduce > 1
        for sub_batch_idx in range(0, len(patch_data), args_.batch_size):

            sub_patch = patch_data[i:i + args_.batch_size]
            batch = sub_patch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))

            # todo: make sure the files in the batch are associated with the right indicies in the relation_mat
            # filename contains sample_id:
            sample_ids_batch = [int(os.path.basename(fn).split('.')[0]) for fn in patch_fn]
            # Relation (adjacent frame, same trajectory)
            if relation_mat is not None:
                batch_relation_mat = relation_mat[sample_ids_batch][:, sample_ids_batch]
                batch_relation_mat = batch_relation_mat.todense()
                # batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None

            # Reconstruction mask
            # recon_loss is computed only on those regions within the supplied mask
            if mask_loader and mask is not None:
                sub_mask = mask_data[i:i + args_.batch_size]
                batch_mask = sub_mask[:, 1:2]  # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.) / 2.  # Add a baseline weight
                batch_mask = batch_mask.permute(0, 2, 1, 3, 4).reshape((-1, 1, x_size, y_size))
            else:
                batch_mask = None

            batch = batch.float()
            batch_mask = batch_mask.float()
            # print(f"subpatch dtype = {batch.dtype}")
            # print(f"subpatch dtype = {batch.type()}")

            # providing a sample from the loader, time relation matrix, and corresponding sample from masks
            _, loss_dict = model(batch,
                                 time_matching_mat=batch_relation_mat,
                                 batch_mask=None)
            # Average gradients among sub-batches
            loss_dict['total_loss'].div_(math.ceil(float(len(patch)) / args_.batch_size))
            loss_dict['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            for key, loss in loss_dict.items():
                if not key in mean_loss:
                    mean_loss[key] = []
                mean_loss[key].append(loss)

    # writer.close()
    return mean_loss

# =======================================================================================


def main_worker(args_):

    args_.cuda = not args_.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args_.batch_size * args_.batches_per_allreduce

    hvd.init()
    torch.distributed.init_process_group('nccl', rank=4)

    if args_.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        print(f"this process's hvd rank = {hvd.local_rank()}")
        # torch.cuda.manual_seed(args_.seed)

    # cudnn.benchmark = True

    # # If set > 0, will resume training from a given checkpoint.
    # resume_from_epoch = 0
    # for try_epoch in range(args_.epochs, 0, -1):
    #     if os.path.exists(args_.checkpoint_format.format(epoch=try_epoch)):
    #         resume_from_epoch = try_epoch
    #         break
    #
    # # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # # checkpoints) to other ranks.
    # resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
    #                                   name='resume_from_epoch').item()

    # # Horovod: print logs on the first worker.
    # verbose = 1 if hvd.rank() == 0 else 0
    #
    # # Horovod: write TensorBoard logs on first worker.
    # try:
    #     if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
    #         from torch.utils.tensorboard import SummaryWriter
    #     else:
    #         from tensorboardX import SummaryWriter
    #     os.makedirs(os.path.join(args_.model_output_dir, 'logs'), exist_ok=True)
    #     log_writer = SummaryWriter(os.path.join(args_.model_output_dir, 'logs')) if hvd.rank() == 0 else None
    # except ImportError:
    #     log_writer = None

    ### MODEL CREATION ###

    # create model
    model1 = VQ_VAE(num_inputs=1, weight_matching=0., channel_var=np.ones((1,)))
    model2 = VQ_VAE(num_inputs=1, weight_matching=0.0005, channel_var=np.ones((1,)))

    model1.cuda()
    model2.cuda()

    model1 = torch.nn.parallel.DistributedDataParallel(model1)
    model2 = torch.nn.parallel.DistributedDataParallel(model2)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    if args_.cuda and args_.use_adasum and hvd.nccl_built():
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        lr_scaler = args_.batches_per_allreduce * hvd.local_size()
    elif not args_.use_adasum:
        lr_scaler = args_.batches_per_allreduce * hvd.size()
    else:
        lr_scaler = 1

    # Horovod: scale learning rate by the number of GPUs.
    optimizer1 = t.optim.Adam(model1.parameters(),
                              lr=(args_.base_lr * lr_scaler),
                              betas=(.9, .999))
    optimizer2 = t.optim.Adam(model2.parameters(),
                              lr=(args_.base_lr * lr_scaler),
                              betas=(.9, .999))

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args_.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer1 = hvd.DistributedOptimizer(
        optimizer1, named_parameters=model1.named_parameters(),
        compression=compression,
        backward_passes_per_step=args_.batches_per_allreduce,
        op=hvd.Adasum if args_.use_adasum else hvd.Average)

    optimizer2 = hvd.DistributedOptimizer(
        optimizer2, named_parameters=model2.named_parameters(),
        compression=compression,
        backward_passes_per_step=args_.batches_per_allreduce,
        op=hvd.Adasum if args_.use_adasum else hvd.Average)

    # # Restore from a previous checkpoint, if initial_epoch is specified.
    # # Horovod: restore on the first worker which will broadcast weights to other workers.
    # if resume_from_epoch > 0 and hvd.rank() == 0:
    #     filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    #     checkpoint = torch.load(filepath)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    ### Settings ###
    model_output_dir = args_.model_output_dir
    project_dir = args_.project_dir

    ### Prepare Data ###
    log.info("LOADING FILES")

    # ======= load data using pytorch systems ========
    torch.set_num_threads(4)
    dataset = DatasetFolderWithPaths(
        root=project_dir+"/JUNE"+"/raw_patches",
        loader=npy_loader,
        extensions='.npy'
    )

    dataset_mask = DatasetFolderWithPaths(
        root=project_dir+"/JUNE"+"/raw_masks",
        loader=npy_loader,
        extensions='.npy'
    )

    relation_mat = np.load(os.path.join(project_dir, "JUNE", "raw_patches", "relation_mat.npy"), allow_pickle=True)

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_sampler_mask = torch.utils.data.distributed.DistributedSampler(
        dataset_mask, num_replicas=hvd.size(), rank=hvd.rank())

    os.makedirs(os.path.join(model_output_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, "stage2"), exist_ok=True)

    # =========================================================
    # =========================================================
    log.info("TRAINING: STARTING STAGE 1")

    kwargs = {'num_workers': 4, 'pin_memory': True} if args_.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    train_mask_loader = torch.utils.data.DataLoader(
        dataset_mask, batch_size=allreduce_batch_size,
        sampler=train_sampler_mask, **kwargs)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model1.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer1, root_rank=0)

    output_dir = os.path.join(model_output_dir, "stage1")
    writer = SummaryWriter(output_dir)
    log.info(f"\ttensorboard logs written to {output_dir}")

    for epoch in range(args_.stage1_epochs):
        model1.train()
        train_sampler.set_epoch(epoch)

        mean_loss = train(model1,
                          train_loader,
                          optimizer1,
                          # relation_mat=relation_mat,
                          mask_loader=train_mask_loader,
                          args_=args_
                          )

        for key, loss in mean_loss.items():
            mean_loss[key] = sum(loss) / len(loss) if len(loss) > 0 else -1.
            writer.add_scalar('Loss/' + key, mean_loss[key], epoch)
        writer.flush()
        log.info('\tepoch %d' % epoch)
        log.info('\t'.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in mean_loss.items()]))

        # only master process should save checkpoints.
        if torch.distributed.get_rank() == 0:
            log.info(f'\t saving epoch {epoch}')
            t.save(model1.state_dict(), os.path.join(output_dir, 'model_epoch%d.pt' % epoch))

    writer.close()

    # =========================================================
    # =========================================================
    log.info("TRAINING: STARTING STAGE 2")

    # get the last saved epoch.  on IBM, use max(). on OSX use min()
    # s1_epochs = glob.glob(os.path.join(model_output_dir, "stage1", "/*"))
    s1_epochs = glob.glob(os.path.join(model_output_dir, "stage1") + '/*.pt')
    last_epoch = max(s1_epochs, key=os.path.getctime)
    log.info(f"\tloading last epoch = {last_epoch}")

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=allreduce_batch_size,
                                               sampler=train_sampler)

    train_mask_loader = torch.utils.data.DataLoader(dataset_mask,
                                                    batch_size=allreduce_batch_size,
                                                    sampler=train_sampler_mask)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model2.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer2, root_rank=0)

    output_dir = os.path.join(model_output_dir, "stage2")
    writer = SummaryWriter(output_dir)
    log.info(f"\ttensorboard logs written to {output_dir}")

    model2.load_state_dict(t.load(last_epoch))
    for epoch in range(args_.stage2_epochs):
        model2.train()
        train_sampler.set_epoch(epoch)

        mean_loss = train(model2,
                          train_loader,
                          optimizer2,
                          # relation_mat=relation_mat,
                          mask_loader=train_mask_loader
                          )

        # shuffle samples ids at the end of the epoch
        # if shuffle_data:
        #     np.random.shuffle(sample_ids)
        for key, loss in mean_loss.items():
            mean_loss[key] = sum(loss) / len(loss) if len(loss) > 0 else -1.
            writer.add_scalar('Loss/' + key, mean_loss[key], epoch)
        writer.flush()
        log.info('\tepoch %d' % epoch)
        log.info('\t'.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in mean_loss.items()]))

        if torch.distributed.get_rank() == 0:
            log.info(f'\t saving epoch {epoch}')
            t.save(model2.state_dict(), os.path.join(output_dir, 'model_epoch%d.pt' % epoch))
    writer.close()


### Visualize reconstruction ###
# def enhance(mat, lower_thr, upper_thr):
#     mat = np.clip(mat, lower_thr, upper_thr)
#     mat = (mat - lower_thr)/(upper_thr - lower_thr)
#     return mat
#
# random_inds = np.random.randint(0, len(dataset), (10,))
# for i in random_inds:
#     sample = dataset[i:(i+1)][0].cuda()
#     cv2.imwrite('sample%d_0.png' % i,
#         enhance(sample[0, 0].cpu().data.numpy(), 0., 1.)*255)
#     cv2.imwrite('sample%d_1.png' % i,
#         enhance(sample[0, 1].cpu().data.numpy(), 0., 1.)*255)
#     output = model(sample)[0]
#     cv2.imwrite('sample%d_0_rebuilt.png' % i,
#         enhance(output[0, 0].cpu().data.numpy(), 0., 1.)*255)
#     cv2.imwrite('sample%d_1_rebuilt.png' % i,
#         enhance(output[0, 1].cpu().data.numpy(), 0., 1.)*255)
