import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-12


def weights_init(layer):
    """
    Initializes (in-place) weights of the given torch.nn Module.

    Args:
        layer: torch.nn Module.

    Returns:
        None.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm1d):
        layer.weight.data.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    else:
        return ValueError


class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau, mu=True):
        super(BN_Layer, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)
        self.theta = torch.tensor(0.5, requires_grad=True)

        self.gamma1 = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.theta))
        self.gamma2 = torch.sqrt((1 - self.tau) * torch.sigmoid((-1) * self.theta))

        self.bn = nn.BatchNorm1d(dim_z)

        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):
        x = self.bn(x)
        return x


class TCDiscriminator(nn.Module):
    """
    Discriminator network for estimating total correlation among a set of RVs.
    Adapted from https://github.com/1Konny/FactorVAE
    """
    def __init__(self, s_dim, c_dim, hidden_size):
        super(TCDiscriminator, self).__init__()
        assert (hidden_size % 2) == 0
        self.s_dim = s_dim
        self.c_dim = c_dim

        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, 2)
        )

        self.map_s = nn.Sequential(
            nn.Linear(s_dim, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True)
        )

        self.map_c = nn.Sequential(
            nn.Linear(c_dim, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, spec, comm):
        s = self.map_s(spec)
        c = self.map_c(comm)
        z = torch.cat([s, c], dim=1)
        return self.score(z).squeeze()


class DPoE(nn.Module):
    def __init__(self, img_size, view_num, latent_dims, hidden_dim=256, tc_dim=1000,
                 share=False, tau=.67, gamma=.5, use_cuda=True, poe=True):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32)

        latent_dims : dict
            View-common and View-specific representation dimensions in neural networks.

        hidden_dim : int
            Encoder dimension. e.g., 256

        tau : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU

        poe : bool
            If True uses POE, else MOE
        """
        super(DPoE, self).__init__()
        self.use_cuda = use_cuda

        # Input data parameters
        self.img_size = img_size  # shape(channel, height, weight)
        self.view_num = view_num
        self.num_pixels = []
        for i in range(self.view_num):
            self.num_pixels.append(img_size[1] * img_size[2])  # pixels = height * weight
        self.reshape = (64, 4, 4)  # The required shape size for ConvTranspose2d()

        # Data representation parameters
        self.hidden_dim = hidden_dim  # Hidden dimension of linear layer (fea_dim -> hidden_dim)
        self.latent_dims = latent_dims
        self.tc_dim = tc_dim

        self.is_specific = 'spec' in self.latent_dims  # True
        self.is_common = 'comm' in self.latent_dims  # True
        self.latent_spec_dim = 0
        self.latent_comm_dim = 0
        if self.is_specific:
            self.latent_spec_dim = self.latent_dims['spec']
        if self.is_common:
            self.latent_comm_dim = self.latent_dims['comm']
        self.latent_dim = self.latent_spec_dim + self.latent_comm_dim

        # Model parameters
        self.share = share
        self.temperature = tau
        self.gamma = gamma
        self.poe = poe

        # >> Define neural nets (input -> hidden -> latent -> hidden -> input)
        img2fea_encoder = []
        fea2vec_encoder = []
        vec2fea_decoder = []
        fea2img_decoder = []
        # -> Define 'img2fea' and 'fea2vec' encoders (input -> hidden)
        for i in range(self.view_num):
            # Initial layer
            encoder_layers = [
                nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
            # Add additional layer if (64, 64) images
            if self.img_size[1:] == (64, 64):
                encoder_layers += [
                    nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                    nn.ReLU()
                ]
            elif self.img_size[1:] == (32, 32):
                # (32, 32) images are supported but do not require an extra layer
                pass
            else:
                raise RuntimeError("{} sized images not supported. Only (#, 32, 32) or (#, 64, 64) supported."
                                   " Build your own architecture or reshape the images!".format(img_size))
            # Add final layers
            encoder_layers += [
                nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
            # Build the img2fea encoder
            img2fea_encoder.append(nn.Sequential(*encoder_layers))  # The '*' is to unpack containers.

            # Define a linear layer to map the encoded features into a hidden vector
            fea2vec_layer = nn.Sequential(
                nn.Linear(64 * 4 * 4, self.hidden_dim),
                nn.ReLU()
                # nn.Sigmoid()
            )
            # Build the fea2vec encoder
            fea2vec_encoder.append(fea2vec_layer)

        # Create the model encoder
        if self.share:
            self.img2fea = nn.ModuleList([img2fea_encoder[0]])
            self.fea2vec = nn.ModuleList([fea2vec_encoder[0]])
        else:
            self.img2fea = nn.ModuleList(img2fea_encoder)
            self.fea2vec = nn.ModuleList(fea2vec_encoder)

        # -> Define latent distribution of Product-of-Experts or Mixture-of-Experts (hidden -> latent)
        # > View-specific latent distribution
        # self.bn_mean = BN_Layer(self.latent_spec_dim, self.gamma, mu=True)
        # self.bn_log_vars = BN_Layer(self.latent_spec_dim, self.gamma, mu=False)
        means = []
        log_vars = []
        if self.is_specific:
            for i in range(self.view_num):
                mean_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.latent_spec_dim),
                    BN_Layer(self.latent_spec_dim, self.gamma, mu=True),
                    # nn.BatchNorm1d(self.latent_spec_dim),
                    # nn.Linear(self.latent_spec_dim, self.latent_spec_dim)
                )
                means.append(mean_layer)
                log_var_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.latent_spec_dim),
                    BN_Layer(self.latent_spec_dim, self.gamma, mu=False),
                    # nn.BatchNorm1d(self.latent_spec_dim),
                    # nn.Linear(self.latent_spec_dim, self.latent_spec_dim)
                )
                log_vars.append(log_var_layer)
        self.means = nn.ModuleList(means)
        self.log_vars = nn.ModuleList(log_vars)

        # > View-common latent distribution
        # tc_tuple = []
        # for i in range(self.view_num):
        #     tc_tuple.append(TCDiscriminator(self.latent_spec_dim, self.latent_comm_dim, hidden_size=self.tc_dim))
        # self.tc_discriminator = nn.ModuleList(tc_tuple)

        # # Fusion mode
        # fc_alpha = []
        # if self.is_common:
        #     fusion_layer = nn.Sequential(
        #         nn.Linear(self.hidden_dim * self.view_num, self.latent_comm_dim)
        #     )
        #     fc_alpha.append(fusion_layer)
        # self.fc_alphas = nn.ModuleList(fc_alpha)

        # # POE or MOE mode
        gates = []
        fc_alphas = []
        if self.is_common:
            for i in range(self.view_num):
                fc_alpha_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.latent_comm_dim),
                    nn.BatchNorm1d(self.latent_comm_dim),
                    nn.Linear(self.latent_comm_dim, self.latent_comm_dim)
                )
                fc_alphas.append(fc_alpha_layer)
        #     gate_layer = nn.Sequential(
        #         nn.Linear(self.hidden_dim, 1)
        #     )
        #     gates.append(gate_layer)
        # self.gates = nn.ModuleList(gates)
        self.fc_alphas = nn.ModuleList(fc_alphas)

        # -> Define 'vec2fea' and 'fea2img' decoders (mirror of the encoders, latent -> hidden -> input)
        for i in range(self.view_num):
            # Define a layer to map the latent vectors into feature space (latent -> hidden)
            vec2fea_layer = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64 * 4 * 4),
                nn.ReLU()
            )
            # Build the vec2fea decoder
            vec2fea_decoder.append(vec2fea_layer)

            # Initial layer
            decoder_layers = []
            # Additional decoding layer for (64, 64) images
            if self.img_size[1:] == (64, 64):
                decoder_layers += [
                    nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                    nn.ReLU()
                ]
            # Final layer
            decoder_layers += [
                nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
                nn.Sigmoid()
                # nn.Tanh()
            ]
            # Build the fea2img decoder
            fea2img_decoder.append(nn.Sequential(*decoder_layers))

        # Create the model decoder
        if self.share:
            self.vec2fea = nn.ModuleList([vec2fea_decoder[0]])
            self.fea2img = nn.ModuleList([vec2fea_decoder[0]])
        else:
            self.vec2fea = nn.ModuleList(vec2fea_decoder)
            self.fea2img = nn.ModuleList(fea2img_decoder)

    def encode(self, X):
        """
        Encodes an image into parameters of a latent distribution defined in self.latent_dims.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        batch_size = X[0].size()[0]
        img_to_fea = []
        fea_to_hidden = []
        for i in range(self.view_num):
            # Encode image to hidden features
            net_num = 0 if self.share else i
            # print(X[i].shape)
            img_to_fea.append(self.img2fea[net_num](X[i]))
            fea_to_hidden.append(self.fea2vec[net_num](img_to_fea[i].view(batch_size, -1)))

        fusion = torch.cat(fea_to_hidden, dim=1)
        # Learning latent representation/distribution from hidden representation (hidden -> latent)
        latent_reps = {}
        if self.is_specific:
            for i in range(self.view_num):
                view_idx = 'spec' + str(i+1)
                mean = self.means[i](fea_to_hidden[i])
                logvar = self.log_vars[i](fea_to_hidden[i])
                # mean = self.bn_mean(self.means[i](fea_to_hidden[i]))  # BN+Scale
                # logvar = self.bn_log_vars(self.log_vars[i](fea_to_hidden[i]))  # BN+Scale
                latent_reps[view_idx] = [mean, logvar]
                # latent_reps[view_idx] = [self.means[i](fea_to_hidden[i]), self.log_vars[i](fea_to_hidden[i])]
        if self.is_common:
            # # Fusion mode
            # latent_reps['comm'] = F.softmax(self.fc_alphas[0](fusion), dim=1)

            if self.poe:
                # # Product-of-Experts
                alpha = []
                for i in range(self.view_num):
                    alpha.append(F.softmax(self.fc_alphas[i](fea_to_hidden[i]), dim=1))
                prob = torch.prod(torch.stack(alpha, dim=0), dim=0)
                norm_prob = prob / torch.sum(prob, dim=-1, keepdim=True)
                latent_reps['comm'] = norm_prob
            else:
                # # Mixture-of-Experts
                alpha = []
                for i in range(self.view_num):
                    alpha.append(F.softmax(self.fc_alphas[i](fea_to_hidden[i]), dim=1))
                prob = torch.sum(torch.stack(alpha, dim=0), dim=0)
                norm_prob = prob / torch.sum(prob, dim=-1, keepdim=True)
                latent_reps['comm'] = norm_prob

                # # MOE with Gates
                # gates = []
                # alpha = []
                # for i in range(self.view_num):
                #     gates.append(self.gates[0](fea_to_hidden[i]))
                #     alpha.append(F.softmax(self.fc_alphas[i](fea_to_hidden[i]), dim=1))
                # gates = F.softmax(torch.stack(gates, dim=1), dim=1)
                # alpha = torch.stack(alpha, dim=1)
                # prob = torch.sum(gates * alpha, dim=1)
                # # prob = torch.sum(alpha, dim=1)  # without Gates
                # norm_prob = prob / torch.sum(prob, dim=-1, keepdim=True)
                # latent_reps['comm'] = norm_prob

        return latent_reps

    def reparameterize(self, latent_reps):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_reps : dict
            Dict with keys 'spec' or 'comm' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []
        if self.is_specific:
            for i in range(self.view_num):
                view_idx = 'spec' + str(i+1)
                mean, logvar = latent_reps[view_idx]
                spec_sample = self.sample_normal(mean, logvar)
                latent_sample.append(spec_sample)
        if self.is_common:
            alpha = latent_reps['comm']
            comm_sample = self.sample_gumbel_softmax(alpha)
            latent_sample.append(comm_sample)

        # Concatenate continuous and discrete samples into one large sample
        # return torch.cat(latent_sample, dim=1)
        return latent_sample

    def discriminator(self, latent_reps, dimperm=True):
        """
        dimperm : bool
            whether to permute the individual dimensions of the modality-specific representation. Default: False.
        """
        tc_scores = []
        num_samples = latent_reps[0].shape[0]
        for i in range(self.view_num):
            s_perm = latent_reps[i].clone()  # permute the individual dimensions of the view-specific representation
            if dimperm:
                for dim in range(s_perm.shape[-1]):
                    s_perm[:, dim] = s_perm[torch.randperm(num_samples), dim]
            else:  # batch-wise permutation, keeping dimensions intact
                s_perm = s_perm[torch.randperm(num_samples)]
            score = self.tc_discriminator[i](latent_reps[i], latent_reps[-1])  # latent_reps[-1]: view-common rep
            perm_score = self.tc_discriminator[i](s_perm, latent_reps[-1])
            tc_scores.append([score, perm_score])

        return tc_scores

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax_1(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

    def sample_gumbel_softmax2(self, alpha, hard=True):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        hard : bool
            hard Gumbel softmax trick
        """
        if self.training:
            logits = torch.log(alpha + EPS)
            gumbels1 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            gumbels2 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            samples = (logits + gumbels1 - gumbels2) / self.temperature  # ~Gumbel(logits, tau)
            y_soft = samples.sigmoid()
        else:
            # Reconstruction mode
            logits = alpha
            y_soft = alpha

        if hard:
            # Straight through
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).masked_fill(y_soft > 0.5, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick
            ret = y_soft
        return ret

    def sample_gumbel_softmax(self, alpha, hard=True):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        hard : bool
            hard Gumbel softmax trick
        """
        if self.training:
            logits = torch.log(alpha + EPS)
            # unif = torch.rand(alpha.size())
            # if self.use_cuda:
            #     unif = unif.cuda()
            # gumbels = -torch.log(-torch.log(unif + EPS) + EPS)
            gumbels = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )  # ~Gumbel(0,1)
            samples = (logits + gumbels) / self.temperature  # ~Gumbel(logits, tau)
            y_soft = F.softmax(samples, dim=1)
        else:
            # Reconstruction mode
            logits = alpha
            y_soft = alpha

        if hard:
            # Straight through
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            # y_hard = torch.zeros_like(
            #     logits, memory_format=torch.legacy_contiguous_format
            # ).masked_fill(y_soft > 0.5, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick
            ret = y_soft
        return ret

    def decode(self, latent_samples):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_samples : torch.Tensor list
            Sample from latent distribution.
        """
        features_to_img = []
        for i in range(self.view_num):
            net_num = 0 if self.share else i
            feature = self.vec2fea[net_num](latent_samples[i])
            features_to_img.append(self.fea2img[net_num](feature.view(-1, *self.reshape)))
        return features_to_img[:]

    def forward(self, X):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W), N:batch_size, C:Channel, H:image_height, W:image_width
        """
        # -> encoder
        latent_reps = self.encode(X)
        # print(latent_reps)
        
        # -> reparameterization trick
        latent_samples = self.reparameterize(latent_reps)
        # -> discriminator with total correlation
        # tc_scores = self.discriminator(latent_samples)

        # -> decoder
        latent_rep_viewlist = []
        for i in range(self.view_num):
            latent_rep_viewlist.append(torch.cat([latent_samples[i], latent_samples[-1]], dim=1))
        decoder_outputs = self.decode(latent_rep_viewlist)
        return decoder_outputs, latent_reps, latent_samples
