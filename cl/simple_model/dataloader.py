import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from collections import defaultdict
import random

class PairedContrastiveDataset(Dataset):
    def __init__(self, embeddings_a, labels_a, embeddings_b, labels_b, n_pos=1, n_neg=1, shuffle=True, seed=42):
        """
        embeddings_a: Tensor [N, D] for modality A (e.g. ST)
        labels_a: Tensor [N] with integer class labels
        embeddings_b: Tensor [M, D] for modality B (e.g. histo)
        labels_b: Tensor [M] with integer class labels
        n_pos: number of positive samples to draw per anchor
        n_neg: number of negative samples to draw per anchor
        """
        super().__init__()
        # Convert to tensors
        self.emb_a = torch.tensor(embeddings_a, dtype = torch.float32)
        self.labels_a = torch.tensor(labels_a, dtype = torch.int32)
        self.emb_b = torch.tensor(embeddings_b, dtype = torch.float32)
        self.labels_b = torch.tensor(labels_b, dtype = torch.int32)
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
        self.seed = seed if seed is not None else 0

        assert len(self.emb_a) == len(self.labels_a)
        assert len(self.emb_b) == len(self.labels_b)

        self.data = []  # List of dicts with precomputed sample indices

        # Build class index lookup
        self.class_to_indices_a = defaultdict(list)
        self.class_to_indices_b = defaultdict(list)
        for i, label in enumerate(labels_a.tolist()):
            self.class_to_indices_a[label].append(i)
        for i, label in enumerate(labels_b.tolist()):
            self.class_to_indices_b[label].append(i)

        all_labels = sorted(set(labels_a.tolist()) | set(labels_b.tolist()))

        # Precompute samples for each item in modality A
        for anchor_idx in range(len(self.emb_a)):
            anchor_label = labels_a[anchor_idx].item()
            self.data.append(self._make_sample(
                anchor_idx=anchor_idx,
                anchor_mod='a',
                anchor_label=anchor_label
            ))

        # Precompute samples for each item in modality B
        for anchor_idx in range(len(self.emb_b)):
            anchor_label = labels_b[anchor_idx].item()
            self.data.append(self._make_sample(
                anchor_idx=anchor_idx,
                anchor_mod='b',
                anchor_label=anchor_label
            ))
        if shuffle:
            self.rng.shuffle(self.data)

    def _make_sample(self, anchor_idx, anchor_mod, anchor_label):
        # Determine anchor embedding source
        if anchor_mod == 'a':
            same_class_to_indices = self.class_to_indices_a
            other_class_to_indices = self.class_to_indices_b
        else:
            same_class_to_indices = self.class_to_indices_b
            other_class_to_indices = self.class_to_indices_a

        # Positive samples from same modality (excluding anchor)
        pos_same_mod = [
            idx for idx in same_class_to_indices[anchor_label]
            if idx != anchor_idx
        ]
        pos_same = self.rng.choice(
            pos_same_mod, size=min(self.n_pos - 1, len(pos_same_mod)), replace = False
        ).astype(np.int32)

        # Negative samples from same modality
        neg_same = []
        for label, indices in same_class_to_indices.items():
            if label != anchor_label:
                neg_same.extend(indices)
        neg_same = self.rng.choice(neg_same, size=min(self.n_neg, len(neg_same)), replace=False).astype(np.int32)

        # Positive samples from other modality
        pos_other = self.rng.choice(
            other_class_to_indices[anchor_label],
            size=min(self.n_pos, len(other_class_to_indices[anchor_label])),
            replace=False
        ).astype(np.int32)

        # Negative samples from other modality
        neg_other = []
        for label, indices in other_class_to_indices.items():
            if label != anchor_label:
                neg_other.extend(indices)
        neg_other = self.rng.choice(neg_other, size=min(self.n_neg, len(neg_other)), replace=False).astype(np.int32)

        return {
            'anchor_mod': anchor_mod,
            'anchor_idx': anchor_idx,
            'pos_same': pos_same,
            'neg_same': neg_same,
            'pos_other': pos_other,
            'neg_other': neg_other
        }

    def shuffle(self):
        self.seed += 1  # Advance the seed to get different assignmenty
        self.rng = np.random.default_rng(self.seed)
        self.data = []

        # Recompute samples for each item in modality A
        for anchor_idx in range(len(self.emb_a)):
            anchor_label = self.labels_a[anchor_idx].item()
            self.data.append(self._make_sample(
                anchor_idx=anchor_idx,
                anchor_mod='a',
                anchor_label=anchor_label
            ))

        # Recompute samples for each item in modality B
        for anchor_idx in range(len(self.emb_b)):
            anchor_label = self.labels_b[anchor_idx].item()
            self.data.append(self._make_sample(
                anchor_idx=anchor_idx,
                anchor_mod='b',
                anchor_label=anchor_label
            ))

        # Shuffle the data as a whole
        self.rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Boolean masks: 1 for positive, 0 for negative
        same_mod_is_positive = torch.tensor(
            [1] * (len(sample['pos_same']) + 1) + [0] * len(sample['neg_same']), dtype=torch.bool
        )
        other_mod_is_positive = torch.tensor(
            [1] * len(sample['pos_other']) + [0] * len(sample['neg_other']), dtype=torch.bool
        )

        if sample['anchor_mod'] == 'a':
            anchor = [sample['anchor_idx'], ]
            a_batch = torch.stack([
                self.emb_a[i] for i in np.concatenate((sample['pos_same'], anchor, sample['neg_same']), axis=0) 
            ])
            b_batch = torch.stack([
                self.emb_b[i] for i in np.concatenate((sample['pos_other'], sample['neg_other']), axis=0)
            ])
            a_is_positive = same_mod_is_positive
            b_is_positive = other_mod_is_positive
        else:
            anchor = [sample['anchor_idx'],]
            b_batch = torch.stack([
                self.emb_b[i] for i in np.concatenate((sample['pos_same'], anchor, sample['neg_same']), axis=0)
            ])
            a_batch = torch.stack([
                self.emb_a[i] for i in np.concatenate((sample['pos_other'], sample['neg_other']), axis=0)
            ])  
            b_is_positive = same_mod_is_positive
            a_is_positive = other_mod_is_positive

        return {
            'a_batch': a_batch,
            'a_is_positive': a_is_positive,
            'b_batch': b_batch,
            'b_is_positive': b_is_positive,
        }

def make_set(
    adata_st, adata_histo, n_pos, n_neg, random_seed, st_key = "pca_embedding", histo_key = "uni_pca_95", 
    class_key = "brain_area_onehot", mode = "train", seed = 42
):
    embeddings_a=adata_st.obsm[st_key]
    labels_a=adata_st.obsm[class_key].toarray().nonzero()[-1]
    embeddings_b=adata_histo.obsm[histo_key]
    labels_b=adata_histo.obsm[class_key].toarray().nonzero()[-1]

    # For st, exclude 10 slides from the train set
    val_slides = list(adata_st.obs["brain_section_label"].unique()[:10])
    print("Validation slides", val_slides)
    st_cond = adata_st.obs["brain_section_label"].isin(val_slides).to_numpy()

    # For histo, exclude 20% of the train set
    rng = np.random.default_rng(seed=seed) 
    sample_size = int(embeddings_b.shape[0] / 5)
    sample = rng.choice(embeddings_b.shape[0], size=sample_size, replace=False)
    histo_cond = np.zeros(shape=(embeddings_b.shape[0]), dtype = bool)
    histo_cond[sample] = True

    if mode == "train":
        _st_cond = ~st_cond
        _histo_cond = ~histo_cond
    elif mode == "val": 
        _st_cond = st_cond
        _histo_cond = histo_cond
    return PairedContrastiveDataset(
        embeddings_a=embeddings_a[_st_cond], 
        labels_a=labels_a[_st_cond], 
        embeddings_b=embeddings_b[_histo_cond], 
        labels_b=labels_b[_histo_cond], 
        n_pos=n_pos,
        n_neg=n_neg,
        seed=random_seed
    )

class CoordinateDataset(Dataset):
    def __init__(self, embeddings, coordinates):
        assert embeddings.shape[0] == coordinates.shape[0]
        self.embeddings = torch.tensor(embeddings, dtype = torch.float32)
        self.coordinates = torch.tensor(coordinates, dtype = torch.float32)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.coordinates[idx]

class DualModalityDataset(Dataset):
    def __init__(self, embeddings_a, embeddings_b, labels_a, labels_b, seed=42):
        self.embeddings_a = torch.tensor(embeddings_a, dtype = torch.float32)
        self.labels_a = torch.tensor(labels_a, dtype = torch.int32)
        self.embeddings_b = torch.tensor(embeddings_b, dtype = torch.float32)
        self.labels_b = torch.tensor(labels_b, dtype = torch.int32)

        self.class_to_indices_a = self._index(labels_a)
        self.class_to_indices_b = self._index(labels_b)
        self.classes = list(set(self.class_to_indices_a) | set(self.class_to_indices_b))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.shuffle_counter = 0

    def _index(self, labels):
        index = defaultdict(list)
        for i, label in enumerate(labels):
            index[int(label)].append(i)
        return index

    def reseed(self):
        self.seed += 1
        self.rng = np.random.default_rng(self.seed)

    def get_class_samples(self, cls, k):
        a_idxs = self.class_to_indices_a.get(cls, [])
        b_idxs = self.class_to_indices_b.get(cls, [])

        a_sample = self.rng.choice(a_idxs, min(k, len(a_idxs)), replace=False) if a_idxs else []
        b_sample = self.rng.choice(b_idxs, min(k, len(b_idxs)), replace=False) if b_idxs else []

        return (
            torch.stack([self.embeddings_a[i] for i in a_sample]) if len(a_sample) > 0 else None,
            torch.stack([self.embeddings_b[i] for i in b_sample]) if len(b_sample) > 0 else None,
            [cls] * len(a_sample),
            [cls] * len(b_sample)
        )

class FixedQuotaSampler(IterableDataset):
    def __init__(self, dataset, class_skew_dict, C=10, K=50, total_samples=100_000, seed = 42):
        self.dataset = dataset
        self.class_skew_dict = class_skew_dict  # class_id -> number of samples per epoch
        self.C = C
        self.K = K
        self.minibatch = C * K
        self.total_samples = total_samples
        self.seed = seed
        self.shuffle_counter = 0
        self.rng = np.random.default_rng(seed)

        self._prepare_epoch_class_queue()

    def shuffle(self):
        self.shuffle_counter += 1
        self.rng = np.random.default_rng(self.seed + self.shuffle_counter)
        self.dataset.reseed()
        self._prepare_epoch_class_queue()

    def _prepare_epoch_class_queue(self):
        # Compute number of times each class must be used in batches
        classes = np.array(list(self.class_skew_dict.keys()))
        classes_to_index = {value: idx for idx, value in enumerate(classes)}
        class_counts = np.array(list(self.class_skew_dict.values()))
        enough_classes_left = True
        class_queue = []

        while(enough_classes_left):            
            classes_i = self.rng.choice(
                classes, size = self.C, replace=False, p=class_counts/class_counts.sum(), shuffle=True
            )
            indices = np.array([classes_to_index[c] for c in classes_i])
            class_counts[indices] = np.maximum(class_counts[indices] - self.K, 0)
            if (class_counts > 0).sum() < self.C:
                # Not enough values left to sample, finish the epoch
                enough_classes_left=False
            else:
                class_queue.append(classes_i)  # We get 2*K samples per class, K per modality

        self.class_queue = np.hstack(class_queue)

    def __iter__(self):
        for i in range(0, len(self.class_queue), self.C):
            selected = self.class_queue[i:self.C + i]
            assert np.unique(selected).size == selected.size

            a_batch, b_batch, a_labels, b_labels = [], [], [], []
            for cls in selected:
                a, b, a_lbl, b_lbl = self.dataset.get_class_samples(cls, self.K)
                if a is not None:
                    a_batch.append(a)
                    a_labels.extend(a_lbl)
                if b is not None:
                    b_batch.append(b)
                    b_labels.extend(b_lbl)

            yield {
                "a_batch": torch.cat(a_batch) if a_batch else torch.empty(0),
                "b_batch": torch.cat(b_batch) if b_batch else torch.empty(0),
                "a_classes": torch.tensor(a_labels),
                "b_classes": torch.tensor(b_labels)
            }
def redistribute_around_mean(arr, scale=0.0):
    arr = np.array(arr, dtype=np.float64)
    mean = np.mean(arr)
    total = np.sum(arr)

    # Compute how far each element is from the mean
    deviation = arr - mean

    # Apply scaling: higher deviations change more
    adjusted = arr - scale * deviation

    # Renormalize to match the original sum
    adjusted_sum = np.sum(adjusted)
    adjusted = adjusted * (total / adjusted_sum)
    adjusted = adjusted.astype(int)

    return adjusted
    
def make_sampler(
    adata_st, adata_histo, C, K, oversample_fraction=0.3, st_key = "pca_embedding", histo_key = "uni_pca_95", 
    class_key = "brain_area_onehot", mode = "train", seed = 42
):
    embeddings_a=adata_st.obsm[st_key]
    labels_a=adata_st.obsm[class_key].toarray().nonzero()[-1]
    embeddings_b=adata_histo.obsm[histo_key]
    labels_b=adata_histo.obsm[class_key].toarray().nonzero()[-1]

    # Split by train/val
    st_cond = adata_st.obs[f"{mode}_set"].to_numpy()
    histo_cond = adata_histo.obs[f"{mode}_set"].to_numpy()
    embeddings_a = embeddings_a[st_cond]
    embeddings_b = embeddings_b[histo_cond]
    labels_a = labels_a[st_cond]
    labels_b = labels_b[histo_cond]

    print(f"ST shape: {embeddings_a.shape}, Histo shape: {embeddings_b.shape}")

    # Get class histogram
    classes, freqs = np.unique(np.hstack([labels_a, labels_b]), return_counts = True)
    classes = classes.astype(int)
    new_freqs = redistribute_around_mean(freqs, oversample_fraction)
    diff = freqs.sum() - new_freqs.sum()
    print("samples lost from class rebalancing:", diff)
    new_freqs[np.argmax(new_freqs)] += diff

    class_counts = dict(zip(classes, new_freqs))
    dataset = DualModalityDataset(embeddings_a, embeddings_b, labels_a, labels_b)
    sampler = FixedQuotaSampler(dataset, class_skew_dict=class_counts, C=C, K=K, seed=seed)
    return sampler