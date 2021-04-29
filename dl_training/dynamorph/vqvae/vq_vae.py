import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

import pdb

CHANNEL_VAR = np.array([1., 1.])
CHANNEL_MAX = 65535.
eps = 1e-9

log = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """ Vector Quantizer module as introduced in 
        "Neural Discrete Representation Learning"

    This module contains a list of trainable embedding vectors, during training 
    and inference encodings of inputs will find their closest resemblance
    in this list, which will be reassembled as quantized encodings (decoder 
    input)

    """
    def __init__(self, embedding_dim=128, num_embeddings=128, commitment_cost=0.25, device=None):
        """ Initialize the module

        Args:
            embedding_dim (int, optional): size of embedding vector
            num_embeddings (int, optional): number of embedding vectors
            commitment_cost (float, optional): balance between latent losses
            device (str, optional): device the model will be running on

        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.device = device if device else None
        self.w = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        """ Forward pass

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: quantized encodings (decoder input)
            torch tensor: quantization loss
            torch tensor: perplexity, measuring coverage of embedding vectors

        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)

        # Decoder input
        encoding_indices = t.argmax(-distances, 1)
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        assert quantized.shape == inputs.shape
        output_quantized = inputs + (quantized - inputs).detach()

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Perplexity (used to monitor)
        if self.device is not None:
            encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings).to(self.device)
        else:
            encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings).cuda()
        encoding_onehot.scatter_(1, encoding_indices.flatten().unsqueeze(1), 1)
        avg_probs = t.mean(encoding_onehot, 0)
        perplexity = t.exp(-t.sum(avg_probs*t.log(avg_probs + 1e-10)))

        return output_quantized, loss, perplexity

    @property
    def embeddings(self):
        return self.w.weight

    def encode_inputs(self, inputs):
        """ Find closest embedding vector combinations of input encodings

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: index tensor of embedding vectors
            
        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)
        encoding_indices = t.argmax(-distances, 1)
        return encoding_indices

    def decode_inputs(self, encoding_indices):
        """ Assemble embedding vector index to quantized encodings

        Args:
            encoding_indices (torch tensor): index tensor of embedding vectors

        Returns:
            torch tensor: quantized encodings (decoder input)
            
        """
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        return quantized


class ResidualBlock(nn.Module):
    """ Customized residual block in network
    """
    def __init__(self,
                 num_hiddens=128,
                 num_residual_hiddens=512,
                 num_residual_layers=2):
        """ Initialize the module

        Args:
            num_hiddens (int, optional): number of hidden units
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers

        """
        super(ResidualBlock, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self.layers = []
        for _ in range(self.num_residual_layers):
            self.layers.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.num_hiddens, self.num_residual_hiddens, 3, padding=1),
                nn.BatchNorm2d(self.num_residual_hiddens),
                nn.ReLU(),
                nn.Conv2d(self.num_residual_hiddens, self.num_hiddens, 1),
                nn.BatchNorm2d(self.num_hiddens)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """ Forward pass

        Args:
            x (torch tensor): input tensor

        Returns:
            torch tensor: output tensor

        """
        output = x
        for i in range(self.num_residual_layers):
            output = output + self.layers[i](output)
        return output


class VQ_VAE(nn.Module):
    """ Vector-Quantized VAE as introduced in 
        "Neural Discrete Representation Learning"
    """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 num_embeddings=64,
                 commitment_cost=0.25,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_commitment=1.,
                 weight_matching=0.005,
                 device=None,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent 
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            num_embeddings (int, optional): number of VQ embedding vectors
            commitment_cost (float, optional): balance between latent losses
            channel_var (list of float, optional): each channel's SD, used for 
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_commitment (float, optional): balance of commitment loss
            weight_matching (float, optional): balance of matching loss
            device (str, optional): device the model will be running on
            **kwargs: other keyword arguments

        """
        super(VQ_VAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_commitment = weight_commitment
        self.weight_matching = weight_matching
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost, device=device)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
      
    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given, 
                pairwise relationship between samples in the minibatch, used 
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask 
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_after, c_loss, perplexity = self.vq(z_before)
        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none') / self.channel_var)
        total_loss = self.weight_recon * recon_loss + self.weight_commitment * c_loss
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_before.reshape((z_before.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = (sim_mat * time_matching_mat).sum()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss,
                'commitment_loss': c_loss,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': perplexity}

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)