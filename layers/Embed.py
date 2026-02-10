import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Patch embedding layer for time series data.
    
    Splits the input time series into fixed-size patches and projects
    them into a d_model-dimensional space.
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Patch embedding via 1D convolution
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Embedding(512, d_model)
        self.dropout = nn.Dropout(dropout)

        # Padding to ensure full coverage
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, n_vars, seq_len)
        Returns:
            x: Patch embeddings of shape (batch_size * n_vars, num_patches, d_model)
            n_vars: Number of variables
        """
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)

        # Unfold into patches: (batch, n_vars, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Reshape: (batch * n_vars, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Project patches
        x = self.value_embedding(x)

        # Add positional embedding
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        return self.dropout(x), n_vars
