from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 编码器：把图片压缩成 低维特征
        self.fc = nn.Sequential(
        # 28x28 -> 784
        
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, latent_dim)  # 输出压缩特征
        )
        
    def forward(self, x):
        z = self.fc(x)  # 编码
        return z
    
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 解码器：把低维特征 还原成图片
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, input_dim),
        )

    def forward(self, z):
        recon_x = self.fc(z)  # 解码
        return recon_x
    

class SigmoidClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, num_classes),
            # nn.BatchNorm1d(num_classes),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(num_classes, num_classes),
            # nn.BatchNorm1d(num_classes),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(num_classes, num_classes),
            # nn.BatchNorm1d(num_classes),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(num_classes, num_classes),
            nn.Sigmoid()
            # nn.BatchNorm1d(num_classes),
        )
        
    def forward(self, fps):
        # z = torch.hstack((fps, r0s))
        out = self.fc(fps)
        # 训练时不需要手动 softmax！CrossEntropyLoss 自带
        return out
    

class SoftmaxClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_classes+latent_dim+3, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(num_classes, num_classes),
            # nn.BatchNorm1d(num_classes)
        )
        
    def forward(self, fps):
        out = self.fc(fps)
        # 训练时不需要手动 softmax！CrossEntropyLoss 自带
        return out
    

class LinearModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.sigmoid_classifier = SigmoidClassifier(latent_dim, num_classes)
        self.softmax_classifier = SoftmaxClassifier(latent_dim, num_classes)
        
    def get_site_prob(self, fps):
        z = self.encoder(fps)
        sig_out = self.sigmoid_classifier(z)
        # 训练时不需要手动 softmax！CrossEntropyLoss 自带
        return sig_out


    def forward(self, fps, r0, r0s):
        z = self.encoder(fps)
        x_recon = self.decoder(z)
        sig_out = self.sigmoid_classifier(z)
        out = torch.hstack((sig_out, z, r0, r0s))
        soft_out = self.softmax_classifier(out)
        # 训练时不需要手动 softmax！CrossEntropyLoss 自带
        return x_recon, sig_out, soft_out