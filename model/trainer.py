"""
main file including scMUSCL implementation

Author: Arash Khoeini
Email: akhoeini@sfu.ca
"""
from re import S
import torch
import numpy as np
import pandas as pd
from model.net import Net
from anndata import AnnData
from typing import List
from model.utils import init_data_loaders, euclidean_dist
from torch.utils.tensorboard import SummaryWriter
from model.metrics import compute_scores
from model.simclr import ContrastiveLoss
from data.experiment import Experiment
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:

    def __init__(self, source_data: List[Experiment], target_data: Experiment, n_clusters: int, config, device: str):

        self.config = config

        if self.config.tensorboard:
            self.writer = SummaryWriter()
        self.genes = source_data[0].genes
        self.n_source_clusters, self.label_dict_reversed = self.prepare_labels(
            source_data, target_data)
        self.source_loaders, self.target_loader, self.pretrain_loader, self.val_loader = \
            init_data_loaders(source_data, target_data, self.config.shuffle_data, 1)
        self.input_dim = len(self.genes)

        self.net = Net(self.input_dim, config.z_dim, config.h_dim,  0)
        self.n_target_clusters = n_clusters
        self.pretrain_epoch = config.epochs_pretrain
        self.device = device
        self.net.to(self.device)

    def prepare_labels(self, source_data: List[Experiment], target_data: Experiment) -> int:
        """
        Counts the number of clusters in all the source datasets combined.
        :param source_data:
        :return: number of clusters
        """
        source_labels = set()
        for src in source_data:
            source_labels.update(src.y)
        target_labels = set(target_data.y)
        labels = []
        labels.extend(list(source_labels))
        labels.extend(list(target_labels - source_labels))
        labels_dict = {v: i for i, v in enumerate(list(labels))}
        labels_dict_reversed = {i: v for v, i in labels_dict.items()}
        for src in source_data:
            src.y = np.array(list(map(lambda x: labels_dict[x], src.y)))

        target_data.y = np.array(
            list(map(lambda x: labels_dict[x], target_data.y)))
        return len(source_labels), labels_dict_reversed

    def initilize_centroids(self):

        with torch.no_grad():
            source_encoded = []
            source_labels = []
            train_iter = [iter(dl) for dl in self.source_loaders]
            for tr_iter in train_iter:
                x, y, _ = next(tr_iter)
                x = x.to(self.device)
                source_encoded.append(self.net.encoder(x).cpu())
                source_labels.append(y)
            source_encoded = torch.cat(source_encoded)
            source_labels = torch.cat(source_labels)

            centroids_train = torch.zeros(
                self.n_source_clusters, self.config.z_dim)
            centroids_test = torch.zeros(self.n_target_clusters, self.config.z_dim)
        
            uniq = torch.unique(source_labels, sorted=True)
            for label in uniq:
                centroids_train[label] = source_encoded[source_labels == label].mean(
                    0)
            
            centroids_test = torch.clone(centroids_train.detach())

            # x, _, _ = next(iter(self.target_loader))
            # x = x.to(self.device)
            # tgt_encoded = self.net.encoder(x).cpu()
            # kmeans = KMeans(n_clusters=self.n_source_clusters, random_state=0).fit(tgt_encoded)
            # kmeans_centers = kmeans.cluster_centers_
            # [centroids_test[i].copy_(torch.Tensor(kmeans_centers[i,:])) for i in range(kmeans_centers.shape[0])]
            
            
        return centroids_train, centroids_test

    def pretrain(self) -> float:
        """
        Pretrains the network
        """
        if self.config.verbose:
            print("Pretraining the network...")
        self.net.train()

        optim = torch.optim.Adam(params=list(
            self.net.parameters()), lr=self.config.learning_rate_pretrain)

        epoch_loss = []
        pbar = tqdm(range(self.config.epochs_pretrain))
        for epoch in pbar:
            total_loss = 0
            for x, _, _ in self.pretrain_loader:
                x = x.to(self.device)
                criterion = ContrastiveLoss(len(x))
                x_hat = self.add_noise(
                    x, ratio=self.config.simclr_noise).to(self.device)
                h = self.net.encoder(x)
                h_hat = self.net.encoder(x_hat)
                loss = criterion(h, h_hat)
                with torch.no_grad():
                    total_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            pbar.set_postfix(pretrain_loss=total_loss)
            epoch_loss.append(total_loss)

        return epoch_loss

    def train(self):
        # Pretraining the network first.
        if self.config.pretrain:
            self.pretrain()

        train_iter = [iter(dl) for dl in self.source_loaders]
        if self.val_loader is not None:
            val_iter = [iter(dl) for dl in self.val_loader]
        test_iter = iter(self.target_loader)

        self.source_centroids, self.target_centroids = self.initilize_centroids()
        self.source_centroids, self.target_centroids = \
            self.source_centroids.to(
                self.device), self.target_centroids.to(self.device)
        
        self.net.train()

        optim = torch.optim.Adam(params=list(
            self.net.encoder.parameters()), lr=self.config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                       gamma=self.config.lr_scheduler_gamma,
                                                       step_size=self.config.lr_scheduler_step)

        optim_centroid = torch.optim.Adam(
            params=[self.target_centroids, self.source_centroids], lr= self.config.learning_rate)
     
        lr_scheduler_cent = torch.optim.lr_scheduler.StepLR(optimizer=optim_centroid,
                                                                   gamma=self.config.lr_scheduler_gamma,
                                                                   step_size=self.config.lr_scheduler_step)

        pbar = tqdm(range(self.config.epochs))
        tgt_ari = 0
        src_ari = 0
        for iteration in pbar:

            ###################Training w.r.t neural network parameters###################
            self.net.train()
            self.net.requires_grad_(True)
            self.target_centroids.requires_grad_(False)
            self.source_centroids.requires_grad_(False)
            # Computing NN loss using source datasets
            nn_loss = 0
            for i, source_loader in enumerate(train_iter):
                x, y, _ = next(source_loader)
                x, y = x.to(self.device), y.to(self.device)
                encoded = self.net.encoder(x)
                nn_loss += (1/len(train_iter)+1) * self.annotated_loss(encoded, y)

            # Computing NN loss using the target dataset
            x, ـ,  ـ = next(test_iter)
            x = x.to(self.device)
            encoded = self.net.encoder(x)
            nn_loss += (1/len(train_iter)+1) * self.unannotated_loss(encoded)

            optim.zero_grad()
            nn_loss.backward()
            optim.step()
            
            ###################Training w.r.t cluster representatives###################
            self.net.requires_grad_(False)
            self.target_centroids.requires_grad_(True)
            self.source_centroids.requires_grad_(True)
            # Computing centroid loss using source datasets
            cent_loss = 0
            for i, source_loader in enumerate(train_iter):
                x, y, _ = next(source_loader)
                x = x.to(self.device)
                y = y.to(self.device)
                encoded = self.net.encoder(x)
                cent_loss +=  (1/(len(train_iter)+1)) * \
                    self.annotated_loss(encoded, y)
            
            # Computing centroid loss using the target dataset
            x, y, _ = next(test_iter)
            x, y = x.to(self.device), y.to(self.device)
            encoded = self.net.encoder(x)
            cent_loss += (1/(len(train_iter)+1)) * self.unannotated_loss(encoded)

            cal_loss = torch.Tensor([0])
            if self.config.cluster_alignment:
                cal_loss = self.cluster_alignment_loss(
                    self.source_centroids, self.target_centroids, iteration)
                cent_loss +=  cal_loss
            
            
            optim_centroid.zero_grad()
            cent_loss.backward()
            optim_centroid.step()
            
            
            if self.config.tensorboard:
                with torch.no_grad():
                    self.writer.add_scalar(
                        'loss/nn', nn_loss.item(), iteration)
                    self.writer.add_scalar(
                         'loss/centroids', cent_loss.item(), iteration)
                self.writer.flush()
            
            if (iteration % self.config.eval_freq) == 0:
                with torch.no_grad():
                    _, src_eval = self.assign_clusters(domain='src')
                    _, tgt_eval = self.assign_clusters(domain='tgt')
                src_ari = src_eval['adj_rand']
                tgt_ari = tgt_eval['adj_rand']
            with torch.no_grad():
                
                pbar.set_postfix(loss= nn_loss.item() + cent_loss.item(),
                                CAL_loss=cal_loss.item(), 
                                src_ari=src_ari, tgt_ari=tgt_ari)
            
            lr_scheduler.step()
            lr_scheduler_cent.step()
        with torch.no_grad():
            adata, eval_results = self.assign_clusters()
            _, src_result = self.assign_clusters(domain='src')
    
        return adata, eval_results, src_result

    def add_noise(self, X: torch.Tensor, ratio=0.05):
        X_hat = X.detach().cpu().clone().numpy()
        nonzero = np.where(X_hat != 0)
        k = int(round(ratio * len(nonzero[0])))
        idx = np.random.choice(list(range(len(nonzero[0]))), k)
        X_hat[(nonzero[0][idx], nonzero[1][idx])] = 0
        return torch.Tensor(X_hat)

    def assign_clusters(self, domain='tgt', evaluation_mode=True):
        self.net.eval()
        self.target_centroids.requires_grad = False
        self.source_centroids.requires_grad = False

        if domain == 'tgt':
            X, labels, cell_ids = next(iter(self.target_loader))
            X = X.to(self.device)
            encoded = self.net.encoder(X)

            dists = euclidean_dist(encoded.cpu(), self.target_centroids.cpu())

            y_pred = torch.min(dists, 1)[1]

            orig_labels = torch.Tensor([self.label_dict_reversed[l.item()] for l in labels])
            orig_y_pred = torch.Tensor([self.label_dict_reversed[l.item()] for l in y_pred])

            adata = self.pack_anndata(X, cell_ids, encoded, orig_labels, orig_y_pred)

            eval_results = None
            if evaluation_mode:
                eval_results = compute_scores(labels, y_pred)

            return adata, eval_results
        elif domain == 'src':
            eval_results = []
            adatas = []
            for src_loader in self.source_loaders:
                X, labels, cell_ids = next(iter(src_loader))
                X = X.to(self.device)
                encoded = self.net.encoder(X)

                dists = euclidean_dist(
                    encoded.cpu(), self.source_centroids.cpu())

                y_pred = torch.min(dists, 1)[1]

                orig_labels = torch.Tensor([self.label_dict_reversed[l.item()] for l in labels])
                orig_y_pred = torch.Tensor([self.label_dict_reversed[l.item()] for l in y_pred])

                adatas.append(self.pack_anndata(X, cell_ids, encoded, orig_labels, orig_y_pred))
                
                if evaluation_mode:
                    eval_results.append(compute_scores(labels, y_pred))
            mean_result = {}
            for k in eval_results[0].keys():
                mean_result[k] = 0
                for i in range(len(eval_results)):
                    mean_result[k] += eval_results[i][k]
                mean_result[k] /= len(eval_results)
            return adatas, mean_result

    def pack_anndata(self, x_input, cells, embedding=None, gtruth=[], estimated=[]):
        """
        Pack results in anndata object.
        """
        adata = AnnData(x_input.data.cpu().numpy())
        adata.obs_names = cells
        adata.var_names = self.genes
        if len(estimated) != 0:
            adata.obs['scGRC_labels'] = pd.Categorical(
                values=estimated.cpu().numpy())
        if len(gtruth) != 0:
            adata.obs['truth_labels'] = pd.Categorical(
                values=gtruth.cpu().numpy())
        if embedding is not None:
            adata.uns['scGRC_embedding'] = embedding.data.cpu().numpy()

        return adata

    def annotated_loss(self, encoded, y):
        """
        Computes loss for annotated cells. Loss is the distance between a cell and it's ground truth cluster centroid.
        :param encoded:
        :param y:
        :return:
        """
        dists = euclidean_dist(encoded, self.source_centroids)
        uniq_y = y.unique()

        loss_val = torch.stack(
            [dists[y == idx_class, idx_class].mean(0) for idx_class in uniq_y]).mean()

        return loss_val.to(self.device)

    def unannotated_loss(self, encoded):
        """
        Computes loss for unannotated cells. Loss is the distance between a cell and it's nearest cluster centroid.
        :param encoded:
        :param y:
        :return:
        """
        dists = euclidean_dist(encoded, self.target_centroids)
        dists = torch.min(dists, axis=1)
        y_hat = dists[1]
        dists = dists[0]
        args_uniq = torch.unique(y_hat, sorted=True)
        loss_val = torch.stack([dists[y_hat == idx_class].mean(0)
                               for idx_class in args_uniq]).mean()

        return loss_val.to(self.device)

    def intercluster_loss(self, centroids):
        dists = euclidean_dist(centroids, centroids)
        sims = 1/ (1+dists)
        nproto = centroids.shape[0]
        loss_val =  torch.sum(sims) / (nproto * nproto - nproto)
        return loss_val

    def cluster_alignment_loss(self, source_centroids, target_centroids,iter , tao=1):
        n_target_centroids = target_centroids.shape[0]
        normalized_concated = F.normalize(torch.concat( [target_centroids, source_centroids]))
        p = 1 / (1e-5+euclidean_dist(normalized_concated,normalized_concated)) / tao
        p[:n_target_centroids,:n_target_centroids].data.fill_(0)
        p[n_target_centroids:, n_target_centroids:].data.fill_(0)
        p = F.softmax(p)
        loss = -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))
        
        
        return loss

    def get_centroids(self):
        return self.source_centroids.cpu().numpy(), self.target_centroids.cpu().numpy()
