import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=True):
        super().__init__()

        stride = 2 if downsample else 1
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)
        return out


class EmotionCRNN(nn.Module):
    def __init__(self, num_outputs=3):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualCNNBlock(1, 32, kernel_size=5, downsample=True),     # -> (32, H/2, W/2)
            ResidualCNNBlock(32, 64, downsample=True),     # -> (64, H/4, W/4)
            ResidualCNNBlock(64, 128, downsample=True),    # -> (128, H/8, W/8)
            ResidualCNNBlock(128, 256, downsample=True),   # -> (256, H/16, W/16)
        )
        
        self.gru_input_size = 256 * 8

        self.proj = nn.Linear(self.gru_input_size, 256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),   # valence, arousal, dominance
            nn.Tanh()
        )

    def forward(self, x):
        """ x shape: [batch, 1, mel_bins, time] """

        x = self.cnn(x)  # Вихід: [B, 256, H_new, T_new]

        B, C, H, W = x.shape 
        x = x.permute(0, 3, 1, 2).contiguous() # [B, T/16, 256, 8]
        x = x.view(B, W, C * H)  # Тепер x має форму [B, T/16, 256*8]

        x = self.proj(x)

        rnn_out, _ = self.gru(x)  
        
        attn_output, _ = self.attention(rnn_out, rnn_out, rnn_out)
        out = attn_output.mean(dim=1)
        out = self.fc(out)

        return out
