import torch
import torch.nn as nn

class sketch_transformer(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_layers=4, num_classes=11, dropout=0.3):
        super(sketch_transformer, self).__init__()

        self.input_proj = nn.Linear(3, d_model)

        self.pos_embed = nn.Parameter(torch.randn(1, 2000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask):
        B, T, _ = x.shape

        x = self.input_proj(x)

        x += self.pos_embed[:, :T, :]

        pad_mask = ~mask.bool()

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.dropout(x)

        masked_sum = (x*mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        pooled = masked_sum/lengths

        logits = self.classifier(pooled)

        return logits