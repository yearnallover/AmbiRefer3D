import torch
import torch.nn as nn
import lightning.pytorch as pl


class CLIPWordEncoder(pl.LightningModule):
    def __init__(self, clip_model, output_channel, dropout):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channel, output_channel),
        )
        self.text_projection = nn.Parameter(
            torch.empty(
                size=(self.clip_model.visual.output_dim, output_channel),
                device=self.device,
                dtype=torch.float32
            )
        )
        self._weight_initialization(output_channel)

    def _weight_initialization(self, output_channel):
        nn.init.normal_(self.text_projection, std=output_channel**-0.5)

    def forward(self, data_dict, output_dict):
        clip_tokens = data_dict["clip_tokens"].flatten(start_dim=0, end_dim=1)
        
        # 1. 严格按照你找出的官方 encode_text 逻辑执行
        x = self.clip_model.token_embedding(clip_tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 获取 word_features
        word_features = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # 获取 sentence_features (这里乘 text_projection)
        eos_token_features = word_features[torch.arange(word_features.shape[0]), clip_tokens.argmax(dim=-1)]
        sentence_features = eos_token_features @ self.clip_model.text_projection

        # 2. 执行后续逻辑
        # 这一步 word_features 是 [Batch, 77, 512]，进入 mlp 处理成 128 维
        word_features = nn.functional.normalize(word_features, dim=2)
        output_dict["word_features"] = self.mlp(word_features)
        
        # 这一步 sentence_features 已经是投影过的，再通过你的 text_projection 矩阵对齐
        output_dict["sentence_features"] = sentence_features @ self.text_projection