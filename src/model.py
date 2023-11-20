    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F 
    # variables


cnn = 3
rnn = 5
transformers = 4
head = 4
hidden_dim = 64
dropout = 0.1
dmodel = 128
class freiqattention(nn.Module):
    def __init__(self, d_model, num_freq_bands):        
        super(freiqattention, self).__init__()
        self.freq_embedding = nn.Emedding(num_freq_bands, d_model) 

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x, freq_indices):
    # x: (batch_size, seq_len, d_model)
    # freq_indices: (batch_size, seq_len)
        freq_embedded = self.freq_embedding(freq_indices)

    # query, key, value: (batch_size, seq_len, d_model)
        query = self.query(x + freq_embedded) 
        key = self.key(x)
        value = self.value(x)
# dot product attention
        A = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        A = torch.softmax(A, dim=-1) 
        O = torch.matmul(A, value) 
        return O

class dyn_pos_enc(nn.Module):
    def __init__(self, d_model, max_len):
        super(dyn_pos_enc, self).__init__()
        

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):        
        super(convblock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding) 
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvAudioTransformer(nn.Module):  
    def __init__(self, input_dim, num_classes, d_model=256, nhead=4, num_encoder_layers=2):
       super().__init__()

        # conv layers
        self.conv1 = convblock(1, 32, kernel_size(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv2 = convblock(32, 64, kernel_size(3, 3), stride=(1, 1), padding=(1, 1)) 

        # transformer layers
        # classes wip                        
        #
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out + self.skip(x)
        return F.relu(out)
class AcousticModel(nn.Module):
    def __init__(self, n_input, n_channel, num_res_blocks=1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(2)

        # Allow for multiple residual blocks
        self.resblocks = nn.Sequential(*[ResBlock(n_channel, n_channel) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.resblocks(x)
        return x

class RecurrentLayers(nn.Module):
    def __init__(self, n_channel, hidden_dim, num_gru_layers=1):
        super().__init__()
        self.gru = nn.GRU(n_channel, hidden_dim, num_layers=num_gru_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.gru(x)
        return x
    
class Transformerblock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads)  # multi-head attention layer
        self.norm1 = nn.LayerNorm(d_model) # layer normalization
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )   # feed-forward network

    def forward(self, x):
        # For self-attention, query, key, and value are all the same
        attended, _ = self.attention(x, x, x)  
        x = self.norm1(attended + x) # residual connection and layer normalization
        fedforward = self.feed_forward(x) 
        return self.norm2(fedforward + x) # residual connection and layer normalization

class StackedTransformer(nn.Module):
    def __init__(self, d_model, heads, N):
        super(StackedTransformer, self).__init__()
        self.layers = nn.ModuleList([Transformerblock(d_model, heads) for _ in range(N)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class nueraspeechASR(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=16, hidden_dim=64, dropout_rate=0.4, 
                 num_res_blocks=cnn, num_gru_layers=rnn, d_model=dmodel, heads=head, num_transformer_layers=transformers):
        super().__init__()


        self.acoustic_model = AcousticModel(n_input, n_channel, num_res_blocks=num_res_blocks) 
        self.dim_matching_conv = nn.Conv1d(n_channel, d_model, kernel_size=1)
        self.transformer_to_rnn_fc = nn.Linear(d_model, n_channel)  # n_channel is 16 in this context
        self.recurrent_layers = RecurrentLayers(n_channel, hidden_dim, num_gru_layers=num_gru_layers) 
        self.attention = StackedTransformer(d_model=hidden_dim * 2, heads=heads, N=num_transformer_layers)
        self.output_fc = nn.Linear(hidden_dim*2, n_output)
    
        self.dropout = nn.Dropout(dropout_rate)  
    def forward(self, x):
        x = self.acoustic_model(x)
        x = self.dim_matching_conv(x)
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x = self.transformer_to_rnn_fc(x)
        x = self.recurrent_layers(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.output_fc(x) 
        return F.log_softmax(x, dim=1)


# Testing the model
n_input = 128
n_output = 35

model = nueraspeechASR(n_input=n_input, n_output=n_output)
print(model)

# Print number of parameters
print(f'The model has {sum(p.numel() for p in model.parameters()):,} trainable parameters')

# Test
x = torch.randn(1, 128, 8000)
OUT = model(x)
print(OUT.shape)
