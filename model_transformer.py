import torch
import torch.nn as nn
import torch.optim as optim

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_decoder_layers, output_seq_len):
        super(TransformerTimeSeries, self).__init__()
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len

        # Input Embedding: 将 21 维初始参数映射到 embed_dim
        self.input_embedding = nn.Linear(input_dim, embed_dim)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_dim))  # max_seq_len=100

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output Projection: 将 embed_dim 映射到单一数值
        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, initial_params):
        """
        initial_params: 初始参数，形状为 [batch_size, 21]
        """
        batch_size = initial_params.size(0)

        # Step 1: Embedding 输入初始参数
        embedded_params = self.input_embedding(initial_params).unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Step 2: 初始化解码器输入序列（全零）
        decoder_input = torch.zeros(batch_size, 1, self.embed_dim, device=initial_params.device)

        # Step 3: 自回归生成序列
        outputs = []
        for t in range(self.output_seq_len):
            # 加位置编码
            decoder_input += self.positional_encoding[:, :decoder_input.size(1), :]

            # 调用解码器
            decoded = self.decoder(
                tgt=decoder_input.permute(1, 0, 2),  # [seq_len, batch_size, embed_dim]
                memory=embedded_params.permute(1, 0, 2)  # [1, batch_size, embed_dim]
            ).permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]

            # 生成当前时间步的预测值
            next_step = self.output_projection(decoded[:, -1, :])  # [batch_size, 1]
            outputs.append(next_step)

            # 将预测值拼接到解码器输入
            decoder_input = torch.cat([decoder_input, decoded[:, -1:, :]], dim=1)

        # 拼接所有时间步的预测值
        outputs = torch.cat(outputs, dim=1)  # [batch_size, output_seq_len]
        return outputs


