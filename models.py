import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {device}')
## empirically slower than CPU on M1
# elif torch.backends.mps.is_built():
#     device = torch.device('mps')
#     print('Using Metal')
else:
    device = torch.device('cpu')
    print('Using CPU')

class InputEncoder(nn.Module):
   
    def __init__(self):
        super().__init__()
        
        self.nonlin = nn.ELU()
        
        # (1, 28, 28) -> (256, 4, 4)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
        ])
        
    def forward(self, x):
        h = x.view(-1, 1, 28, 28)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.nonlin(h)
        return h

# mu, logvar = q(c | h_1,...,h_m)
class StatisticNetwork(nn.Module):
    
    def __init__(
        self,
        batch_size=16, sample_size=5,
        n_features=256*4*4, hidden_dim=256, c_dim=512,
    ):
        super().__init__()
        
        self.batch_size = batch_size        # datasets/batch
        self.sample_size = sample_size      # examples/dataset 
        
        self.n_features = n_features        # output of encoder
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
       
        self.nonlin = nn.ELU()
        
        self.pre_fc = nn.Linear(self.n_features, self.hidden_dim)
        self.pre_batch = nn.BatchNorm1d(self.hidden_dim) 
        
        self.post_fc_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2*self.c_dim)
        ])
        
        self.post_batch_layers = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.BatchNorm1d(1),
        ])

    def forward(self, h):
        h = h.view(-1, self.n_features)
        
        h = self.pre_fc(h)
        h = self.pre_batch(h)
        h = self.nonlin(h)
        
        h = self.pool(h)
        
        # the last batch norm is "global"
        for i, (fc, bn) in enumerate(zip(self.post_fc_layers, self.post_batch_layers)):
            h = fc(h)
            if i < len(self.post_fc_layers) - 1:
                h = bn(h)
                h = self.nonlin(h)
            else:
                h = h.view(-1, 1, 2*self.c_dim)
                h = bn(h)
                h = h.view(-1, 2*self.c_dim)
        
        return h[:, :self.c_dim], h[:, self.c_dim:]
       
    def pool(self, h):
        h = h.view(self.batch_size, self.sample_size, self.hidden_dim)
        return h.mean(1)
    
    # this implemented previously, should try to ablate this? 
    def dropout(self, h):
        a = torch.ones((self.batch_size, 1, 1)).to(device)
        b = torch.bernoulli(([1, 0.5][self.training]) * torch.ones((self.batch_size, self.sample_size-1, 1))).to(device)
        mask = torch.cat([a, b], 1)
        
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e * mask.expand_as(e)               # dropout samples

        kept = torch.sum(mask, 1)               # (batch_size, 1)
        e = torch.sum(e, 1)                     # (batch_size, 1, hidden_dim)
        e /= kept.expand_as(e)                  # (batch_size, 1, hidden_dim)

        e = torch.cat([e, kept], 2).squeeze(1)  # (batch_size, hidden_dim+1)

        return e
    
class ResidualNetwork(nn.Module):
    
    def __init__(self, hidden_dim=256, n_hidden=3):
        super().__init__()
        self.nonlin = nn.ELU()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ])
        ])
        
    def forward(self, x):
        init_x = x # should be a copy here implicitly
        for i, [fc, bn] in enumerate(self.layers):
            x = fc(x)
            x = bn(x)
            if i < len(self.layers) - 1: x = self.nonlin(x)
            else: x = self.nonlin(x + init_x)
        return x

# mu_z, s2_z = p(z_i-1 | z_i, h, c)   
class InferenceNetwork(nn.Module):
    
    def __init__(
        self,
        batch_size=16, sample_size=5,
        n_features=256*4*4, hidden_dim=256, c_dim=512, z_dim=16,
        n_hidden=3
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_hidden = n_hidden
       
        self.nonlin = nn.ELU()
        self.fc_h = nn.Linear(self.n_features, self.hidden_dim) 
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim) 
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim) 
        self.res_net = ResidualNetwork(self.hidden_dim, self.n_hidden)
        self.post_fc = nn.Linear(self.hidden_dim, 2*self.z_dim)
        self.post_bn = nn.BatchNorm1d(1)
        
    def forward(self, z, h, c):
        
        # if there's a previous hidden state
        if z is not None:
            z = z.view(-1, self.z_dim)
            z = self.fc_z(z)
            z = z.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            z = torch.zeros(self.batch_size, self.sample_size, self.hidden_dim).to(device)
        
        h = h.view(-1, self.n_features)
        h = self.fc_h(h)
        h = h.view(self.batch_size, self.sample_size, self.hidden_dim)
        
        c = self.fc_c(c)
        c = c.view(self.batch_size, 1, self.hidden_dim).expand_as(h)

        # concate and feed to resnet
        o = z + h + c
        o = o.view(-1, self.hidden_dim)
        o = self.nonlin(o)
        o = self.res_net(o)
        
        # "global" and extract params
        o = self.post_fc(o)
        o = o.view(-1, 1, 2*self.z_dim)
        o = self.post_bn(o)
        o = o.view(-1, 2*self.z_dim)
       
        # note to self: these are diagonal gaussian parameters
        return o[:, :self.z_dim], o[:, self.z_dim:]

# mu_z, s2_z = p(z_i-1 | z_i, c) 
class LatentDecoder(nn.Module):
    
    def __init__(
        self,
        batch_size=16, sample_size=5,
        n_features=256*4*4, hidden_dim=256, c_dim=512, z_dim=16,
        n_hidden=3 
    ):
        super().__init__()        
        
        self.batch_size = batch_size 
        self.sample_size = sample_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim 
        self.n_hidden = n_hidden
       
        self.nonlin = nn.ELU() 
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.res_net = ResidualNetwork(self.hidden_dim, self.n_hidden)
        self.post_fc = nn.Linear(self.hidden_dim, 2*self.z_dim)
        self.post_bn = nn.BatchNorm1d(1)
        
    def forward(self, z, c):
        # basically same as with inference network
        if z is not None:
            z = z.view(-1, self.z_dim)
            z = self.fc_z(z)
            z = z.view(self.batch_size, self.sample_size, self.hidden_dim) 
        else:
            z = torch.zeros(self.batch_size, self.sample_size, self.hidden_dim).to(device)
            
        c = self.fc_c(c)
        c = c.view(self.batch_size, 1, self.hidden_dim).expand_as(z)
        
        o = z + c
        o = o.view(-1, self.hidden_dim)
        o = self.nonlin(o)
        o = self.res_net(o)
        
        o = self.post_fc(o) 
        o = o.view(-1, 1, 2*self.z_dim)
        o = self.post_bn(o)
        o = o.view(-1, 2*self.z_dim)
        
        return o[:, :self.z_dim], o[:, self.z_dim:]
        
# p(x | z, c)
class ObservationDecoder(nn.Module):
    
    def __init__(
        self,
        batch_size=16, sample_size=5,
        n_features=256*4*4, hidden_dim=256, c_dim=512, z_dim=16,
        n_latent=3
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.n_latent = n_latent
       
        self.nonlin = nn.ELU() 
        self.z_concat_fc = nn.Linear(self.n_latent * self.z_dim, self.hidden_dim)
        self.c_fc = nn.Linear(self.c_dim, self.hidden_dim)
        self.project_fc = nn.Linear(self.hidden_dim, self.n_features)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1), # padding 0 here for right shape
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        ])
        
        self.deproject_conv = nn.Conv2d(64, 1, kernel_size=1)
       
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
        ])
        
    def forward(self, z_concat, c):
        z_concat = self.z_concat_fc(z_concat)
        z_concat = z_concat.view(self.batch_size, self.sample_size, self.hidden_dim)
        
        c = self.c_fc(c)
        c = c.view(self.batch_size, 1, self.hidden_dim).expand_as(z_concat)
       
        x = self.nonlin(z_concat + c)
        x.view(-1, self.hidden_dim)
        
        x = self.project_fc(x)
        x = x.view(-1, self.hidden_dim, 4, 4)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.nonlin(x)
            
        x = self.deproject_conv(x)
        x = F.sigmoid(x)
        
        return x
    
class NeuralStatistician(nn.Module):
    
    def __init__(self):
        super().__init__() 
        
        self.encoder = InputEncoder()
        self.statistic_network = StatisticNetwork()
        self.inference_networks = nn.ModuleList([InferenceNetwork() for _ in range(3)])
        self.latent_decoders = nn.ModuleList([LatentDecoder() for _ in range(3)])
        self.observation_decoder = ObservationDecoder()
        
        self.apply(self.init_weights) # from https://github.com/conormdurkan/neural-statistician
       
    @staticmethod 
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)
    
    # VAE reparameterization trick, not too familiar 
    def reparam(self, mu, logvar):
        stdev = torch.exp(0.5 * logvar)
        eps = torch.randn(stdev.shape).to(device)
        return mu + stdev * eps
            
    def forward(self, x):
        
        # (bsz, ssz, 256, 4, 4)
        h = self.encoder(x) # (bsz, ssz, 256, 4, 4)

        # (16, 512), (16, 512)
        mu_c, logvar_c = self.statistic_network(h)
        c = self.reparam(mu_c, logvar_c)
        
        q_samples = []
        q_params, p_params = [], []
        
        prev_z = None
        for i, layer in enumerate(self.inference_networks):
            mu_z, logvar_z = layer(prev_z, h, c)
            q_params.append((mu_z, logvar_z))
            z = self.reparam(mu_z, logvar_z)
            q_samples.append(z)
            prev_z = z
        
        prev_z = None    
        for i, layer in enumerate(self.latent_decoders):
            mu_z, logvar_z = layer(prev_z, c)
            p_params.append((mu_z, logvar_z))
            prev_z = q_samples[i]
            
        print([q.shape for q in q_samples])
        
        # (bsz*ssz, n_latent*z_dim) 
        z_concat = torch.cat(q_samples, dim=1)
        print(z_concat.shape)
        px = self.observation_decoder(z_concat, c)
        
        return mu_c, logvar_c, q_samples, q_params, p_params, px, x
    
    def loss(self, outputs, weight):
        pass