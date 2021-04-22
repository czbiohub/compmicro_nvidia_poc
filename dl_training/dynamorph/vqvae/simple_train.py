import numpy as np
import os
import pickle
# from .vq_vae_supp import reorder_with_trajectories, vae_preprocess
from dl_training.dynamorph.vqvae.vq_vae_supp import reorder_with_trajectories, vae_preprocess
#import vq_vae_supp.reorder_with_trajectories as reorder_with_trajectories
# import vq_vae_supp.vae_preprocess as vae_preprocess
from .vq_vae import VQ_VAE
# import vq_vae.VQ_VAE as VQ_VAE
import torch as t
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import logging
import glob


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler()
#     ]
# )
log = logging.getLogger(__name__)


def train(model,
          dataset,
          output_dir,
          use_channels=[],
          relation_mat=None,
          mask=None,
          n_epochs=10,
          lr=0.001,
          batch_size=16,
          device='cuda:0',
          shuffle_data=False,
          transform=True,
          seed=None):
    """ Train function for VQ-VAE, VAE, IWAE, etc.

    Args:
        model (nn.Module): autoencoder model
        dataset (TensorDataset): dataset of training inputs
        output_dir (str): path for writing model saves and loss curves
        use_channels (list, optional): list of channel indices used for model
            training, by default all channels will be used
        relation_mat (scipy csr matrix or None, optional): if given, sparse
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        batch_size (int, optional): batch size
        device (str, optional): device (cuda or cpu) where models are running
        shuffle_data (bool, optional): shuffle data at the end of the epoch to
            add randomness to mini-batch; Set False when using matching loss
        transform (bool, optional): data augmentation
        seed (int, optional): random seed

    Returns:
        nn.Module: trained model

    """
    if not seed is None:
        np.random.seed(seed)
        t.manual_seed(seed)
    total_channels, n_z, x_size, y_size = dataset[0][0].shape[-4:]
    if len(use_channels) == 0:
        use_channels = list(range(total_channels))
    n_channels = len(use_channels)
    assert n_channels == model.num_inputs

    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()

    n_samples = len(dataset)
    n_batches = int(np.ceil(n_samples / batch_size))
    # Declare sample indices and do an initial shuffle
    sample_ids = np.arange(n_samples)
    if shuffle_data:
        np.random.shuffle(sample_ids)
    writer = SummaryWriter(output_dir)
    log.info(f"\ttensorboard logs written to {output_dir}")

    for epoch in range(n_epochs):
        mean_loss = {'recon_loss': [],
                     'commitment_loss': [],
                     'time_matching_loss': [],
                     'total_loss': [],
                     'perplexity': []}
        log.info('\tstart epoch %d' % epoch)
        for i in range(n_batches):
            # Deal with last batch might < batch size
            sample_ids_batch = sample_ids[i * batch_size:min((i + 1) * batch_size, n_samples)]
            batch = dataset[sample_ids_batch][0]
            assert len(batch.shape) == 5, "Input should be formatted as (batch, c, z, x, y)"
            batch = batch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))
            batch = batch.to(device)

            # Data augmentation
            if transform:
                for idx_in_batch in range(len(sample_ids_batch)):
                    img = batch[idx_in_batch]
                    flip_idx = np.random.choice([0, 1, 2])
                    if flip_idx != 0:
                        img = t.flip(img, dims=(flip_idx,))
                    rot_idx = int(np.random.choice([0, 1, 2, 3]))
                    batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])

            # Relation (adjacent frame, same trajectory)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[sample_ids_batch][:, sample_ids_batch]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None

            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[sample_ids_batch][0][:, 1:2]  # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.) / 2.  # Add a baseline weight
                batch_mask = batch_mask.permute(0, 2, 1, 3, 4).reshape((-1, 1, x_size, y_size))
                batch_mask = batch_mask.to(device)
            else:
                batch_mask = None

            _, loss_dict = model(batch,
                                 time_matching_mat=batch_relation_mat,
                                 batch_mask=batch_mask)
            loss_dict['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            for key, loss in loss_dict.items():
                if not key in mean_loss:
                    mean_loss[key] = []
                mean_loss[key].append(loss)
        # shuffle samples ids at the end of the epoch
        if shuffle_data:
            np.random.shuffle(sample_ids)
        for key, loss in mean_loss.items():
            mean_loss[key] = sum(loss) / len(loss) if len(loss) > 0 else -1.
            writer.add_scalar('Loss/' + key, mean_loss[key], epoch)
        writer.flush()
        log.info('\tepoch %d' % epoch)
        log.info('\t'.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in mean_loss.items()]))

        # log.info(f"\tcheckpoint save at epoch {epoch}")
        t.save(model.state_dict(), os.path.join(output_dir, 'model_epoch%d.pt' % epoch))

    writer.close()
    return model


def main(args_):

    ### Settings ###
    channels = args_.channels
    model_output_dir = args_.model_output_dir
    device = args_.device
    project_dir = args_.project_dir

    # channels = [1]
    # model_output_dir = "./retardance_only_model"
    # device = "cuda:1"

    ### Prepare Data ###
    log.info("LOADING FILES")
    fs = pickle.load(open(os.path.join(project_dir, 'JUNE', 'raw', 'D_file_paths.pkl'), 'rb'))
    dataset = pickle.load(open(os.path.join(project_dir, 'JUNE', 'raw', 'D_static_patches.pkl'), 'rb'))
    dataset_mask = pickle.load(open(os.path.join(project_dir, 'JUNE', 'raw', 'D_static_patches_mask.pkl'), 'rb'))
    relations = pickle.load(open(os.path.join(project_dir, 'JUNE', 'raw', 'D_static_patches_relations.pkl'), 'rb'))

    # path = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation'
    # fs = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_file_paths.pkl'), 'rb'))
    # dataset = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches.pkl'), 'rb'))
    # dataset_mask = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches_mask.pkl'), 'rb'))
    # relations = pickle.load(open(os.path.join(path, 'JUNE', 'raw', 'D_static_patches_relations.pkl'), 'rb'))

    # Reorder
    log.info("PREPARING LOADED DATA")
    dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
    fs = [fs[i] for i in inds_in_order]
    dataset_mask = dataset_mask[np.array(inds_in_order)]
    dataset = vae_preprocess(dataset, use_channels=channels)

    dataset = TensorDataset(t.from_numpy(dataset).float())
    dataset_mask = TensorDataset(t.from_numpy(dataset_mask).float())

    os.makedirs(os.path.join(model_output_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, "stage2"), exist_ok=True)

    # Stage 1 training
    log.info("TRAINING: STARTING STAGE 1")
    model = VQ_VAE(num_inputs=1, alpha=0., channel_var=np.ones((1,)), device=device)
    model = model.to(device)
    model = train(model,
                  dataset,
                  os.path.join(model_output_dir, "stage1"),
                  relation_mat=relation_mat,
                  mask=dataset_mask,
                  # n_epochs=100,
                  n_epochs=10,
                  lr=0.0001,
                  batch_size=128,
                  device=device,
                  shuffle_data=False,
                  transform=True)

    log.info("TRAINING: STARTING STAGE 2")
    model = VQ_VAE(num_inputs=1, alpha=0.0005, channel_var=np.ones((1,)), device=device)
    model = model.to(device)
    # get the last saved epoch.  on IBM, use max(). on OSX use min()
    # s1_epochs = glob.glob(os.path.join(model_output_dir, "stage1", "/*"))
    s1_epochs = glob.glob(os.path.join(model_output_dir, "stage1") + '/*.pt')
    last_epoch = max(s1_epochs, key=os.path.getctime)
    # model.load_state_dict(t.load(os.path.join(model_output_dir, "stage1", "model_epoch99.pt")))
    model.load_state_dict(t.load(last_epoch))
    model = train(model,
                  dataset,
                  os.path.join(model_output_dir, "stage2"),
                  relation_mat=relation_mat,
                  mask=dataset_mask,
                  # n_epochs=400,
                  n_epochs=40,
                  lr=0.0001,
                  batch_size=128,
                  device=device,
                  shuffle_data=False,
                  transform=True)
    pass


### Check coverage of embedding vectors ###
# used_indices = []
# for i in range(500):
#     sample = dataset[i:(i+1)][0].cuda()
#     z_before = model.enc(sample)
#     indices = model.vq.encode_inputs(z_before)
#     used_indices.append(np.unique(indices.cpu().data.numpy()))
# print(np.unique(np.concatenate(used_indices)))

### Generate latent vectors ###
# z_bs = {}
# z_as = {}
# for i in range(len(dataset)):
#     sample = dataset[i:(i+1)][0].cuda()
#     z_b = model.enc(sample)
#     z_a, _, _ = model.vq(z_b)
#     f_n = fs[inds_in_order[i]]
#     z_as[f_n] = z_a.cpu().data.numpy()
#     z_bs[f_n] = z_b.cpu().data.numpy()
  

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
