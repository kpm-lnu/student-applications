import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)  # residual connection
        out = self.pool(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim**0.5

    def forward(self, x):
        # x: [batch, time, hidden]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        out = attn @ V  # weighted sum
        return out


class EmotionCRNN(nn.Module):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualCNNBlock(1, 32),      # -> (32, H/2, W/2)
            ResidualCNNBlock(32, 64),     # -> (64, H/4, W/4)
            ResidualCNNBlock(64, 128),    # -> (128, H/8, W/8)
            ResidualCNNBlock(128, 256),   # -> (256, H/16, W/16)
        )
        self.rnn_input_dim = 256
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.attention = SelfAttention(hidden_dim=512)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),   # valence, arousal
        )

    def forward(self, x):
        """     x shape: [batch, 1, mel_bins, time]     """
        x = self.cnn(x)  # [B, C, H, T]
        B, C, H, T = x.shape
        x = x.mean(dim=2).transpose(1, 2)
        rnn_out, _ = self.gru(x)  # [B, T, 512]
        attn_out = self.attention(rnn_out)
        out = attn_out.mean(dim=1)  # [B, 512]
        out = self.fc(out)
        return out
