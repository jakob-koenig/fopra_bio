import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch.optim import SGD, Adam
from torcheval.metrics.functional import r2_score
import numpy as np
import random

import scanpy as sc
from pytorch_lightning.callbacks import TQDMProgressBar
import os

class L2NormalizationLayer(nn.Module):
    def __init__(self):
        super(L2NormalizationLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class SimpleModel(nn.Module):
    def __init__(self, 
                 in_dim: int = 512, 
                 hidden_dims: list[int] = [512, 256], 
                 out_dim: int = 128,
                 dropout: float = 0.1,
                 use_batchnorm: bool = True,
                 activation: str = 'relu',
                 projection_head: bool = False):
        """
        A simple MLP for projecting embeddings into a shared latent space.
        
        Args:
            in_dim (int): Input feature dimension.
            hidden_dims (list[int]): List of hidden layer dimensions.
            out_dim (int): Final output dimension (latent space size).
            dropout (float): Dropout probability between layers.
            use_batchnorm (bool): Whether to use BatchNorm between layers.
            activation (str): Activation function to use ('relu' or 'gelu').
        """
        super().__init__()
        
        assert activation in ['relu', 'gelu'], "Activation must be 'relu' or 'gelu'"
        act_fn = nn.ReLU if activation == 'relu' else nn.GELU
        
        layers = []
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final projection layer (no activation here)          
        self.projection_head = projection_head
        if projection_head:
            layers.append(L2NormalizationLayer())
            
        layers.append(nn.Linear(dims[-1], out_dim))
        layers.append(L2NormalizationLayer())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def embed(self, x):
        if self.projection_head:
            return self.model[:-2](x)
        else:
            return self.model(x)

class SimCLR(pl.LightningModule):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(
       self, hidden_dims, latent_dim, lr, temperature, weight_decay, loss_function = "SupCon",
       dropout = 0.1, use_batchnorm = True, max_epochs=500, projection_head=False,
       shuffle_every=None, train_set = None, log_every = 2000, histo_size = 368, st_size = 50):
       super().__init__()
       self.save_hyperparameters(ignore=["train_set",])
       assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
       self.histo_model = SimpleModel(
           in_dim = histo_size, hidden_dims = hidden_dims, out_dim = latent_dim, dropout = dropout, 
           use_batchnorm = use_batchnorm, projection_head=projection_head
       )
       self.st_model = SimpleModel(
           in_dim = st_size, hidden_dims = hidden_dims, out_dim = latent_dim, dropout = dropout, 
           use_batchnorm = use_batchnorm, projection_head=projection_head
       )
       self.log_every = log_every
       self.train_dataset = train_set
       self.shuffle_every = shuffle_every
       self.loss_function = loss_function
       assert loss_function in ["SupCon", "SimCLR"]
       self.init_loss_trackers()

   def init_loss_trackers(self):
       self.losses = {"train": [], "val": []}
       self.sims = {"train": dict(pos = [], neg = []), "val" : dict(pos = [], neg = [])}
       self.accs = {"train": dict(top1 = [], top5 = [], mean_pos = []), "val" : dict(top1 = [], top5 = [], mean_pos = [])}

   def on_train_start(self):
       self.logger.log_hyperparams(
           self.hparams,
           metrics={"dummy_metric": 0.0}
       )
       
   def on_train_epoch_end(self):
       for mode in ("train", "val"):
           if len(self.losses[mode]) > 0:
               self.log(f"{mode}_loss_per_epoch", np.mean(self.losses[mode]))
           for key in self.accs[mode].keys():
               if len(self.accs[mode][key]) > 0: 
                   self.log(f"{mode}_acc_{key}_per_epoch", np.mean(self.accs[mode][key]))
           for key in ("pos", "neg"):
               if len(self.sims[mode][key]) > 0: 
                   self.log(f"{mode}_sim_{key}_per_epoch", np.mean(self.sims[mode][key]))
       self.init_loss_trackers()

   def on_validation_epoch_end(self):
       self.on_train_epoch_end()  # Repeat for validation
       

   def on_train_epoch_start(self):
       # Shuffle every N epochs
       if self.train_dataset == None or self.shuffle_every == None: return
       if self.current_epoch % self.shuffle_every == 0 and self.current_epoch > 0:
           if hasattr(self.train_dataset, "shuffle"):
                self.train_dataset.shuffle()
               
   def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

   def info_nce_loss(self, batch, batch_idx, mode = "train"):
        X_st = batch['a_batch'].squeeze(0)
        classes_st = batch['a_is_positive'].squeeze(0)
        X_histo = batch['b_batch'].squeeze(0)
        classes_histo = batch['b_is_positive'].squeeze(0)
       
        # X_st = F.normalize(X_st, p=2, dim=1)
        # X_histo = F.normalize(X_histo, p=2, dim=1)  # MOved normalization to SimpleModel
        # print(f"ST shape: {X_st.shape}, Histo shape: {X_histo.shape}")
        emb_histo = self.histo_model(X_histo)
        emb_st = self.st_model(X_st)
       
        # Index the embeddings
        pos_st = emb_st[classes_st]
        pos_histo = emb_histo[classes_histo]
        neg_st = emb_st[~classes_st]
        neg_histo = emb_histo[~classes_histo]
        
        # Concatenate positives and negatives
        positives = torch.cat([pos_st, pos_histo], dim=0)
        negatives = torch.cat([neg_st, neg_histo], dim=0)
        n_pos = positives.shape[0]
        n_neg = negatives.shape[0]
       
        nce_parts = []
        similarities = []
        features = ((positives, positives), (positives, negatives))
        for i in range(2):
            # Repeat for positives and negatives
            feat1, feat2 = features[i]
            
            cos_sim = F.cosine_similarity(feat1[:,None,:], feat2[None,:,:], dim=-1)
            if i == 0:
                # Remove the diagonal from positive to positive comparison
                self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                cos_sim.masked_fill_(self_mask, 0)  # This will vanish in the logsum exp
            cos_sim = cos_sim / self.hparams.temperature
            if i == 0:
                nce_parts.append(torch.sum(cos_sim, dim=-1) / n_pos)
            else:
                nce_parts.append(torch.logsumexp(cos_sim, dim=-1))
            similarities.append(cos_sim)

       # Now both parts of the loss are of shape (n_pos,)
        
        nll = nce_parts[0] - nce_parts[1]
        nll = -nll.sum()

        # Logging loss
        self.log(mode+'_loss', nll)
        self.losses[mode].append(nll.detach().cpu().numpy())

        # More in depth logging (less frequent)
        if mode == "val" or (batch_idx % self.log_every == 0):
            with torch.no_grad():
                sim_pos = similarities[0].masked_fill(self_mask, 0)
                comb_sim = torch.cat([
                    sim_pos.sum(dim = 0) / (n_pos - 1),  # Mean cosine distance of positive samples
                    similarities[1].mean(dim = 0)  # Again for negative samples
                ], dim = -1)
                sim_argsort = comb_sim.argsort(dim=-1, descending=True) 
                classes = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim = -1).to(sim_argsort.device)

                pos_sim = comb_sim[:n_pos].mean(dim=-1)
                neg_sim = comb_sim[n_pos:].mean(dim=-1)
                self.sims[mode]["pos"].append(pos_sim.cpu().numpy())
                self.sims[mode]["neg"].append(neg_sim.cpu().numpy())
                self.log(mode+'_mean_sim_pos', pos_sim)
                self.log(mode+'_mean_sim_neg', neg_sim)
    
                assert sim_argsort.max() < len(classes)
                top_classes = classes[sim_argsort] 
                acc_top1 = top_classes[0].float()
                acc_top5 = top_classes[:5].float().mean()
                acc_mean_pos = top_classes[:n_pos].float().mean()

                self.accs[mode]["top1"].append(acc_top1.cpu().numpy())
                self.accs[mode]["top5"].append(acc_top5.cpu().numpy())
                self.accs[mode]["mean_pos"].append(acc_mean_pos.cpu().numpy())
                self.log(mode+'_acc_top1', acc_top1)
                self.log(mode+'_acc_top5', acc_top5)
                self.log(mode+'_acc_mean_pos', acc_mean_pos)
            
        return nll

   def supcon_loss(self, batch, batch_idx, mode="train"):
        # Extract input and labels
        X_st = batch['a_batch'].squeeze(0)              
        y_st = batch['a_classes'].squeeze(0)
        
        X_histo = batch['b_batch'].squeeze(0)       
        y_histo = batch['b_classes'].squeeze(0)
        
        # Get normalized embeddings (handled in the models)
        z_st = self.st_model(X_st)       # [N_a, D]
        z_histo = self.histo_model(X_histo)  # [N_b, D]
        
        # Combine embeddings and labels
        embeddings = torch.cat([z_st, z_histo], dim=0)         # [N, D]
        labels = torch.cat([y_st, y_histo], dim=0)             # [N]
        N = embeddings.size(0)
        
        # Cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.hparams.temperature  # [N, N]
        # Uses the dotproduct since it is faster and vectors are normalized automatically
        
        # Mask to remove self-comparisons
        self_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        sim_matrix = sim_matrix.masked_fill(self_mask, -9e15)
        
        # Build positive mask: same class, different index
        labels = labels.contiguous().view(-1, 1)  # [N, 1]
        positive_mask = (labels == labels.T) & ~self_mask  # [N, N], boolean
        
        # Compute log prob
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # [N, N]
        
        # Only average over positives
        pos_mask_sum = positive_mask.sum(1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / pos_mask_sum.clamp(min=1)
        
        # Final loss
        loss = -mean_log_prob_pos.mean()
        
        # Logging
        self.log(f"{mode}_loss", loss, on_epoch=True, prog_bar=True)
        self.losses[mode].append(loss.detach().cpu().numpy())

        # More in depth logging (less frequent)
        if mode == "val" or (batch_idx % self.log_every == 0):
            with torch.no_grad():
                pos_sim = (sim_matrix * positive_mask).sum(1) / pos_mask_sum.clamp(min=1)
                pos_sim = pos_sim[pos_mask_sum > 0]
                negative_mask = ~torch.logical_or(positive_mask, self_mask)
                neg_mask_sum = negative_mask.sum(1)
                neg_sim = (sim_matrix * negative_mask).sum(1) / neg_mask_sum.clamp(min=1)
                neg_sim = neg_sim[neg_mask_sum > 0]

                comb_sim = torch.cat([pos_sim, neg_sim], dim = -1)
                sim_argsort = comb_sim.argsort(dim=-1, descending=True) 
                n_pos = pos_sim.size(0)
                n_neg = neg_sim.size(0)
                classes = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim = -1).to(sim_argsort.device)
                
                pos_sim = pos_sim.mean(dim=-1)
                neg_sim = neg_sim.mean(dim=-1)
                self.sims[mode]["pos"].append(pos_sim.cpu().numpy())
                self.sims[mode]["neg"].append(neg_sim.cpu().numpy())
                self.log(mode+'_mean_sim_pos', pos_sim)
                self.log(mode+'_mean_sim_neg', neg_sim)
    
                assert sim_argsort.max() < len(classes)
                top_classes = classes[sim_argsort] 
                acc_top1 = top_classes[0].float()
                acc_top5 = top_classes[:5].float().mean()
                acc_mean_pos = top_classes[:n_pos].float().mean()

                self.accs[mode]["top1"].append(acc_top1.cpu().numpy())
                self.accs[mode]["top5"].append(acc_top5.cpu().numpy())
                self.accs[mode]["mean_pos"].append(acc_mean_pos.cpu().numpy())
                self.log(mode+'_acc_top1', acc_top1)
                self.log(mode+'_acc_top5', acc_top5)
                self.log(mode+'_acc_mean_pos', acc_mean_pos)
        
        return loss

   def training_step(self, batch, batch_idx):
       if self.loss_function == "SimCLR":
           return self.info_nce_loss(batch, batch_idx, mode='train')
       else:
           return self.supcon_loss(batch, batch_idx, mode='train')

   def validation_step(self, batch, batch_idx):
       if self.loss_function == "SimCLR":
           return self.info_nce_loss(batch, batch_idx, mode='val')
       else:
           return self.supcon_loss(batch, batch_idx, mode='val')

# Model for coordinate regression ================================================
class CoordinateRegressionModule(pl.LightningModule):
    def __init__(self, max_epochs = 100, in_dim=512, hidden_dims=[512, 256], out_dim=3, lr=1e-4, weight_decay = 5e-8, dropout = 0, use_batchnorm=False, activation = "relu"):
        super().__init__()
        self.save_hyperparameters()

        self.model = SimpleModel(
            in_dim=self.hparams.in_dim,
            hidden_dims=self.hparams.hidden_dims,
            out_dim=self.hparams.out_dim,
            dropout = dropout,
            use_batchnorm=use_batchnorm,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
       self.logger.log_hyperparams(
           self.hparams,
           metrics={"dummy_metric": 0.0}
       )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        with torch.no_grad():
            self.log("val_r2", r2_score(y, y_hat))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

def train_simclr(train_dataset, val_dataset, CHECKPOINT_PATH, refresh_rate = 2000, max_epochs=500, loss_function="SupCon",
                 seed = 42, limit_train_batches = None, logger = None, n_workers_fraction = 1.0, monitor = "val_loss",
                 monitor_mode = "min", determinicstic=False, patience = 10, return_monitored=False, **kwargs
    ):
    if loss_function == "SimCLR":
        NUM_WORKERS = int(os.cpu_count() * n_workers_fraction)  # Reserve some workers
    else:
        NUM_WORKERS = 1
    torch.set_float32_matmul_precision('medium')
    print("NUM_WORKERS:", NUM_WORKERS)

    # Setting the seed
    pl.seed_everything(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    if determinicstic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         logger = logger,
                         limit_train_batches=limit_train_batches,  # To only use x% each epoch
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode=monitor_mode, monitor=monitor),
                             ModelCheckpoint(save_weights_only=True, every_n_epochs=1, filename='last'),
                             EarlyStopping(
                                 monitor=monitor, patience=patience, mode=monitor_mode, verbose=True
                             ),
                             LearningRateMonitor('epoch'), progress_bar])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                   pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=NUM_WORKERS)

    model = SimCLR(max_epochs=max_epochs, train_set=train_dataset, loss_function=loss_function, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLR.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        weights_only=True
    ) # Load best checkpoint after training

    if return_monitored:
        best_mon = trainer.callback_metrics.get(monitor)
        if best_mon is not None:
            return model, best_mon.item()
        else:
            best_mon = float("inf") if monitor_mode == "max" else -float("inf")
            return model, best_mon
            
    return model

def train_coordinate_regressor(train_dataset, val_dataset, CHECKPOINT_PATH, batch_size = 64, refresh_rate = 2000,
                               max_epochs=100, seed = 42,  logger = None, n_workers_fraction = 1.0, monitor = "val_r2",
                               determinicstic=False, patience = 8, **kwargs
    ):
    NUM_WORKERS = int(os.cpu_count() * n_workers_fraction)  # Reserve some workers
    print("NUM_WORKERS:", NUM_WORKERS)
    torch.set_float32_matmul_precision('medium')

    # Setting the seed
    pl.seed_everything(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    if determinicstic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         logger = logger,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor=monitor),
                                    ModelCheckpoint(save_weights_only=True, every_n_epochs=1, filename='last'),
                                    EarlyStopping(
                                        monitor=monitor, patience=patience, mode='max', verbose=True
                                    ),
                                    LearningRateMonitor('epoch'), progress_bar])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS
    )
    
    model = CoordinateRegressionModule(out_dim=3, **kwargs)
    trainer.fit(model, train_loader, val_loader)

    model = CoordinateRegressionModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, weights_only=False
    ) # Load best checkpoint after training

    return model
