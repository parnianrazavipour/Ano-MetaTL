from torch import nn
import torch


class AnomalyDetector(nn.Module):
    def __init__(self, k, input_embed_size=100, series_embed_size=200, ano_embed_size=50, dropout_p=0.5):
        super(AnomalyDetector, self).__init__()
        self.k = k
        self.series_fc = nn.Sequential(
            nn.Linear(self.k * input_embed_size, series_embed_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p))

        self.ano_fc = nn.Sequential(
            nn.Linear(input_embed_size, ano_embed_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p))

        self.final_fc = nn.Sequential(
            nn.Linear(series_embed_size + ano_embed_size, 1),
            nn.Sigmoid())

    def forward(self, series, anomaly):
        series_embed = self.series_fc(series)
        anomaly_embed = self.ano_fc(anomaly)
        cat_embed = torch.cat([series_embed, anomaly_embed], 1)
        out = self.final_fc(cat_embed)
        return out
