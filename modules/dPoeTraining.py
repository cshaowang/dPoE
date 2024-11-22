import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm.asyncio import tqdm

EPS = 1e-12


class Trainer(object):
    def __init__(self, model, tc_tuple, optimizer, spec_cap=None, comm_cap=None,
                 record_loss_every=100, view_num=2, datax='dataset', use_cuda=True):
        """
        **Acknowledgments**

        Class to handle training of model.

        Parameters
        ----------
        model : modules.dPoeModel.DPoE instance

        tc_tuple ：tuple, modules.dPoeModel.TCDiscriminator and its optimizer

        optimizer : torch.optim.Optimizer instance

        spec_cap : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the view-specific latent
            channels. Cannot be None if model.is_specific is True.

        comm_cap : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the view-common latent channels.
            Cannot be None if model.is_common is True.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True, pass model to GPU.
        """
        self.model = model
        self.tc_tuple = tc_tuple  # tuple:(tc, tc_optimizer) x view_num
        self.optimizer = optimizer
        self.spec_cap = spec_cap
        self.comm_cap = comm_cap
        self.record_loss_every = record_loss_every
        self.view_num = view_num
        self.datax = datax
        self.use_cuda = use_cuda
        if self.model.is_specific and self.spec_cap is None:
            raise RuntimeError("Model is view-specific but spec_cap not provided.")

        if self.model.is_common and self.comm_cap is None:
            raise RuntimeError("Model is view-common but comm_cap not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.optimizer_steps = 0
        self.view_beta_gain = []
        for i in range(self.view_num):
            self.view_beta_gain.append(1.)
        self.batch_size = None
        self.losses = {'loss': [], 'recon_loss': [], 'kl_loss': []}

        self.loss_r = [[], [], []]  # recon_loss
        self.loss_zv = [[], [], []]  # kl_loss zv (view-specific)
        self.loss_zc = [[], [], []]  # kl_loss zc (view-common)
        self.loss_tc = [[], [], []]  # kl_loss zc (tc discriminator)

        self.mean_loss = [[], [], []]

        # Keep track of divergence values for each latent variables
        if self.model.is_specific:
            self.losses['kl_loss_spec'] = []
            # For every dimension of view-specific latent variables
            for i in range(self.model.latent_dims['spec']):
                self.losses['kl_loss_spec_' + str(i)] = []

        if self.model.is_common:
            self.losses['kl_loss_comm'] = []
            # For every dimension of view-common latent variable
            for i in range(self.model.latent_dims['comm']):
                self.losses['kl_loss_comm_' + str(i)] = []

    def train(self, data_loader, scheduler, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        scheduler : torch.optim.lr_scheduler.StepLR

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.train()  # Sets the model in training mode.
        results_on_zc = []
        results_on_zv = [[], [], []]
        for epoch in range(epochs):
            # print("Epoch %d with learning rate: %f" % (epoch, self.optimizer.param_groups[0]['lr']))
            mean_epoch_loss = self._train_epoch(epoch, data_loader)
            # scheduler.step()  # Decays the learning rate of each parameter group by gamma every step_size epochs.
            # scd_tc.step()  # Decays the learning rate of each parameter group by gamma every step_size epochs.
            for i in range(self.view_num):
                mean_loss_view_i = self.model.num_pixels[i] * mean_epoch_loss[i]
                # print('Average loss view-%d: %.2f' % (i + 1, mean_loss_view_i))
                self.mean_loss[i].append(mean_loss_view_i)

            lookup = False
            if lookup and (epoch % 5 == 0):
                from utils.DataLoader import mv_data_loader
                lookup_data, lookup_views, lookup_clusters, _ = mv_data_loader(data_name=self.datax + '.mat',
                                                                               train_mode=False)
                for batch_idx, batch_data in enumerate(lookup_data):
                    if batch_idx != 0:
                        raise RuntimeError("Lookup data load error. Lookup data should be loaded in one batch.")
                    datax = batch_data[0:-1]
                    labels = batch_data[-1]
                    inputs = []
                    from torch.autograd import Variable
                    for i in range(lookup_views):
                        inputs.append(Variable(datax[i]))
                        inputs[i] = inputs[i].cuda()
                    encodings = self.model.encode(inputs)
                    from utils.evaluation import test
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=lookup_clusters, n_init=100)
                    zc = encodings['comm'][0].cpu().detach().data.numpy()
                    y = labels.cpu().detach().data.numpy()
                    for i in range(lookup_views):
                        view_idx = 'spec' + str(i + 1)
                        zv = encodings[view_idx][0].cpu().detach().data.numpy()  # cz
                        pv = kmeans.fit_predict(zv)
                        results_on_zv[i].append(test(y, pv))
                    pc = kmeans.fit_predict(zc)
                    results_on_zc.append(test(y, pc))
                    # p_max = zc.argmax(1)
                    # acc_cmax = test(y, p_max)
                    # acc_abs = test(pc, p_max)
                    # if acc_abs == 1:
                    #     break

                if save_training_gif is not None:
                    # Generate batch of images and convert to grid
                    viz = save_training_gif[1]
                    viz.save_images = False
                    img_grid = viz.all_latent_traversals(size=10)
                    # Convert to numpy and transpose axes to fit imageio convention
                    # i.e. (width, height, channels)
                    img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                    # Add image grid to training progress
                    training_progress_images.append(img_grid)
        # np.save('./MultiZv.npy', results_on_zv)
        # np.save('./MultiZc.npy', results_on_zc)
        # np.save('./mean_loss.npy', self.mean_loss)

        if save_training_gif is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images, fps=24)

    def _train_epoch(self, epoch, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = []
        for i in range(self.view_num):
            epoch_loss.append(0.)

        with tqdm(data_loader, desc="Epoch %d -lr %1.2e" % (epoch, self.optimizer.param_groups[0]['lr'])) as tqdm_bar:
            for Data in tqdm_bar:
                iter_loss = self._train_iteration(Data[0:-1])  # [:-1] as the last dimension in 'Data' is class label.
                # batch_loss
                batch_loss = 0.
                for i in range(self.view_num):
                    epoch_loss[i] += iter_loss[i]
                    batch_loss += iter_loss[i]
                tqdm_bar.set_postfix(loss=batch_loss)

        epoch_view_loss = []
        for i in range(self.view_num):
            epoch_view_loss.append(epoch_loss[i] / len(data_loader.dataset))  # average view_loss over all data
        return epoch_view_loss

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.optimizer_steps += 1
        for i in range(self.view_num):
            # self.tc_tuple[i][1].zero_grad()
            if self.use_cuda:
                data[i] = data[i].cuda()
        self.tc_tuple[1].zero_grad()
        self.optimizer.zero_grad()
        decoder_outputs, latent_reps, latent_samples = self.model(data)
        recon_data = []
        for i in range(self.view_num):
            recon_data.append(decoder_outputs[i])

        Loss = []

        # the first view
        max_recon_loss = F.binary_cross_entropy(recon_data[0].view(-1, self.model.num_pixels[0]),
                                                data[0].view(-1, self.model.num_pixels[0]))
        max_recon_loss *= self.model.num_pixels[0]
        # print(float(max_recon_loss))
        recon_loss = [max_recon_loss]
        # the second view to the last view
        for i in range(1, self.view_num):
            recon_view_loss = F.binary_cross_entropy(recon_data[i].view(-1, self.model.num_pixels[i]),
                                                     data[i].view(-1, self.model.num_pixels[i]))
            recon_view_loss *= self.model.num_pixels[i]
            # print(float(recon_view_loss))
            recon_loss.append(recon_view_loss)
            if max_recon_loss < recon_view_loss:
                max_recon_loss = recon_view_loss
        # print(self.optimizer_steps)
        # if self.optimizer_steps == 1:
        #     for i in range(self.view_num):
        #         self.view_beta_gain[i] = float(recon_loss[i]) / float(max_recon_loss)
            # print(self.view_beta_gain)

        # print(max_recon_loss)
        # print(latent_reps)
        latent_samples_detached = []
        for i in range(self.view_num):
            latent_samples_detached.append(latent_samples[i].detach())  # latent_samples[i]: view-specific rep
        latent_samples_detached.append(latent_samples[-1].detach())  # latent_samples[-1]: view-common rep

        for i in range(self.view_num):
            view_idx = 'spec' + str(i + 1)
            Loss.append(self._loss_function(data[i], recon_data[i],
                                            {'spec': latent_reps[view_idx], 'comm': latent_reps['comm']},
                                            latent_samples[i], latent_samples[-1],
                                            view=i, max_recon_loss=max_recon_loss, beta_gain=self.view_beta_gain[i]))

        total_loss = Loss[-1]
        for i in range(self.view_num-1):
            total_loss += Loss[i]
        total_loss.backward()
        # for i in range(self.view_num-1):
        #     Loss[i].backward(retain_graph=True)
        # Loss[-1].backward()
        # gradient clipping to avoid exploding gradients, used after .backward() and before .step()
        # clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()
        self.optimizer.zero_grad()

        for i in range(self.view_num):
            # Loss_tc = self._cross_tc_loss(self.tc_tuple[i],
            #                               latent_samples_detached[i], latent_samples_detached[-1], tc_train=True)
            Loss_tc = self._cross_tc_loss(self.tc_tuple,
                                          latent_samples_detached[i], latent_samples_detached[-1], tc_train=True)
            Loss_tc.backward()
            # self.tc_tuple[i][1].step()  # [1]: tc optimizer
            # self.tc_tuple[i][1].zero_grad()
            self.tc_tuple[1].step()  # [1]: tc optimizer
            self.tc_tuple[1].zero_grad()

        train_loss = []
        for i in range(self.view_num):
            train_loss.append(Loss[i].item())
        # print(train_loss)
        return train_loss

    def _loss_function(self, data, recon_data, latent_reps, spec_samples, comm_samples,
                       view=0, max_recon_loss=1000, beta_gain=1.):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_reps : dict
            Dict with keys 'spec' or 'comm' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy
        recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels[view]),
                                            data.view(-1, self.model.num_pixels[view]))  # negative log

        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels[view]
        # print(recon_loss, self.model.num_pixels[view])
        # Calculate KL divergences
        kl_spec_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_comm_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        spec_cap_loss = 0
        comm_cap_loss = 0

        spec_max, spec_beta, spec_delta, max_cap_batch = self.spec_cap
        comm_max, comm_beta, max_cap_batch = self.comm_cap

        spec_step_cap_rate = spec_max / max_cap_batch
        comm_step_cap_rate = comm_max / max_cap_batch

        if self.model.is_specific:
            # Calculate KL divergence
            mean, logvar = latent_reps['spec']
            kl_spec_loss = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of view-specific channels
            # Increase view-specific capacity without exceeding spec_max
            spec_cap_current = spec_step_cap_rate * self.optimizer_steps
            # spec_cap_current = torch.mean(mean).item()
            spec_cap_current = min(spec_cap_current, spec_max)
            # Calculate view-specific capacity loss
            # spec_beta = spec_beta * recon_loss / max_recon_loss
            # spec_beta = spec_beta * beta_gain
            spec_cap_loss = spec_beta * torch.abs(spec_cap_current - kl_spec_loss)
            # spec_cap_loss = spec_beta * kl_spec_loss
        if self.model.is_common:
            # Calculate KL divergence
            kl_comm_loss = self._kl_gumbel_loss(latent_reps['comm'])
            # Linearly increase capacity of view-common channels
            # Increase view-common capacity without exceeding comm_max or theoretical
            comm_cap_current = comm_step_cap_rate * self.optimizer_steps
            comm_cap_current = min(comm_cap_current, comm_max)
            # Calculate view-common capacity loss
            # comm_beta = comm_beta * recon_loss / max_recon_loss
            # comm_beta = comm_beta * beta_gain
            comm_cap_loss = comm_beta * torch.abs(comm_cap_current - kl_comm_loss)
            # comm_cap_loss = comm_beta * kl_comm_loss
        # Calculate total kl value to record it
        # kl_loss = kl_spec_loss + kl_comm_loss
        # Calculate tc loss
        # cross_tc_loss = self._cross_tc_loss(self.tc_tuple[view], spec_samples, comm_samples, tc_train=False)
        cross_tc_loss = self._cross_tc_loss(self.tc_tuple, spec_samples, comm_samples, tc_train=False)
        tc_loss = spec_delta * cross_tc_loss
        # Calculate total loss
        total_loss = recon_loss + spec_cap_loss + comm_cap_loss + tc_loss
        # print(self.optimizer_steps, recon_loss, spec_cap_loss+comm_cap_loss, kl_loss)

        # Record losses
        # if self.model.training and self.optimizer_steps % self.record_loss_every == 0:
        #     self.losses['recon_loss'].append(recon_loss.item())
        #     self.losses['kl_loss'].append(kl_loss.item())
        #     self.losses['loss'].append(total_loss.item())
        #     # print(recon_loss.data, kl_spec_loss.data, kl_comm_loss.data)
        #
        # self.loss_r[view].append(recon_loss.item())
        # self.loss_zv[view].append(kl_spec_loss.item())
        # self.loss_zc[view].append(kl_comm_loss.item())

        # To avoid large losses, so normalizing by the number of pixels
        return total_loss / self.model.num_pixels[view]

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # see Eq.2 in the model of BN-VAE (ACL-2020)
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.optimizer_steps % self.record_loss_every == 1:
            self.losses['kl_loss_spec'].append(kl_loss.item())
            for i in range(self.model.latent_dims['spec']):
                self.losses['kl_loss_spec_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_gumbel_loss(self, alpha):
        """
        Calculates the KL divergence between a set of gumbel experts
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alpha : tensor
            The alpha parameters of a categorical (or gumbel-softmax)
            distribution with shape (N, D).
        """
        # Calculate kl losses for each view-common latent
        # kl_losses = [self._kl_view_comm_loss(alpha) for alpha in alphas]
        comm_dim = int(alpha.size()[-1])
        log_dim = torch.log(torch.tensor(comm_dim))
        # log_dim = torch.Tensor([np.log(comm_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_losses = log_dim + mean_neg_entropy

        # Total loss is sum of kl loss for each view-common latent
        kl_loss = kl_losses  # torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.optimizer_steps % self.record_loss_every == 1:
            self.losses['kl_loss_comm'].append(kl_loss.item())
            # for i in range(len(alphas)):
            #     self.losses['kl_loss_comm_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _cross_tc_loss(self, tc_tuple, spec_samples, comm_samples, tc_train, dimperm=False):
        """
        Estimates total correlation (TC) between a set of variables and optimizes
        the TCDiscriminator if train=true.
        NOTE: adapted from FactorVAE (https://github.com/1Konny/FactorVAE)

        Args:
            spec_samples : view-specific latent samples, tensor
            comm_samples : view-common latent samples, tensor

        Returns:
            tc_loss is the loss of the cross-entropy loss of the discriminator
        """
        tc, tc_opt = tc_tuple
        num_samples = spec_samples.shape[0]
        zeros = torch.zeros(num_samples, dtype=torch.long)
        ones = torch.ones(num_samples, dtype=torch.long)
        if self.use_cuda:
            zeros = zeros.cuda()
            ones = ones.cuda()
        s_perm = spec_samples.clone()  # permute the individual dimensions of the view-specific representation
        if dimperm:
            for dim in range(s_perm.shape[-1]):
                s_perm[:, dim] = s_perm[torch.randperm(num_samples), dim]
        else:  # batch-wise permutation, keeping dimensions intact
            s_perm = s_perm[torch.randperm(num_samples)]

        if tc_train is True:
            # compute the CEL and backprop within the discriminator
            score = tc(spec_samples, comm_samples)
            perm_score = tc(s_perm, comm_samples)
            tc_loss = 0.5 * (F.cross_entropy(score, zeros) + F.cross_entropy(perm_score, ones))
        else:
            # estimate tc
            scores = tc(spec_samples, comm_samples)
            tc_reg = F.log_softmax(scores, dim=1)
            tc_loss = (tc_reg[:, 0] - tc_reg[:, 1]).mean()

        return tc_loss

    def compute_infomax(self, projection_head, h1, h2, tau=1.0):
        """
        Estimates the mutual information between a set of variables.
        Automatically uses $K = batch_size - 1$ negative samples.

        Args:
            projection_head: projection head for the MI-estimator. Can be identity.
            h1: torch.Tensor, first representation
            h2: torch.Tensor, second representation
            tau: temperature hyperparameter.

        Returns:
            A tuple (mi, d_loss) where mi is the estimated mutual information and
            d_loss is the cross-entropy loss computed from contrasting
            true vs. permuted pairs.
        """

        # compute cosine similarity matrix C of size 2N * (2N - 1), w/o diagonal elements
        batch_size = h1.shape[0]
        z1 = projection_head(h1)
        z2 = projection_head(h2)
        z1_normalized = F.normalize(z1, dim=-1)
        z2_normalized = F.normalize(z2, dim=-1)
        z = torch.cat([z1_normalized, z2_normalized], dim=0)  # 2N * D
        C = torch.mm(z, z.t().contiguous())  # 2N * 2N
        # remove diagonal elements from C
        mask = ~ torch.eye(2 * batch_size, device=C.device).type(torch.ByteTensor)  # logical_not on identity matrix
        C = C[mask].view(2 * batch_size, -1)  # 2N * (2N - 1)

        # compute loss
        numerator = 2 * torch.sum(z1_normalized * z2_normalized) / tau
        denominator = torch.logsumexp(C / tau, dim=-1).sum()
        loss = (denominator - numerator) / (2 * batch_size)
        return np.nan, loss  # NOTE: Currently returns MI=NaN
