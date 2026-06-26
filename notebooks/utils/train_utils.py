import abc
import os
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, cast, Union, Iterator
from rtree.index import Index, Property
import rasterio
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Sampler, Dataset
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torchgeo.datasets import RasterDataset, stack_samples, BoundingBox, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips

#os.chdir("D:/githubs/vjepa2/notebooks")
os.chdir(os.path.join(os.getcwd(), ".."))



class GeoSampler(Sampler[BoundingBox], abc.ABC):

    def __init__(self, dataset: GeoDataset, rois: Optional[BoundingBox] = None) -> None:
        if rois is None:
            self.index = dataset.index
            rois = [BoundingBox(*self.index.bounds)]
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            for roi in rois: 
                hits = dataset.index.intersection(tuple(roi), objects=True)
                for hit in hits:
                    bbox = BoundingBox(*hit.bounds) & roi
                    self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.rois = rois

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        pass

class MultipleGridGeoSampler(GeoSampler):
    """
    Returns a grid of Bounding Boxes over each specified ROI
    """
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi_bounds: List[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, None)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = roi_bounds

        self.length = 0
        for hit in self.hits:
            bounds = hit
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        for hit in self.hits:
            bounds = hit
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        return self.length

class MultiDateRGBDataset(RasterDataset):
    is_image = True
    
    def __init__(self, filepaths, band_indices=None, crs = None, res = None, transforms=None, sep_dates=False):

        super().__init__(paths=[filepaths[0]], crs=crs, res=res, transforms=transforms)

        self.paths = filepaths
        self.datasets = [rasterio.open(fp) for fp in filepaths]
        self.band_indices = band_indices if band_indices is not None else None
        self.sep_dates = sep_dates

    def __getitem__(self, query):
        tiles = []
        for path in self.paths:
            tile = self._merge_files([path], query, self.band_indexes)
            if self.band_indices is not None : 
                tile = tile[self.band_indices, :, :]
            tiles.append(tile)
        sample = {"crs": self.crs, "bbox": query}
        tiles = np.stack(tiles, axis=0)  # Shape: (num_dates, num_bands, tile_size, tile_size)
        if not self.sep_dates :
            shp = tiles.shape
            tiles = tiles.reshape(shp[0]*shp[1], *shp[2:]) # Shape: (num_dates * num_bands, tile_size, tile_size)
            sample['image'] = torch.tensor(tiles).to(self.dtype)
            if self.transforms is not None:
                sample = self.transforms(sample)

        else:
            tiles = torch.tensor(tiles).to(self.dtype)
            #sample["image"] = torch.einsum('tchw->cthw', tiles)
            sample["image"] = tiles
            #print(sample['image'].shape)
            if self.transforms is not None:
                sample = self.transforms(sample)
            #print(sample['image'].shape)
            #sample["image"] = torch.einsum('tchw->cthw', sample['image'])
        return sample  
    
class RobustGeoDataset:
    """Skips out_of_bounds_samples"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, query):
        try:
            return self.dataset[query]
        except Exception as e:
            return None
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

class DictTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, sample):
        sample['image'] = self.transforms(sample['image'])
        return sample
    
def polygon_to_bbox(polygon, buffer=None):
    bounds = list(polygon.bounds)
    if buffer is not None:
        if not isinstance(buffer, tuple):
            buffer = (buffer, buffer)
        buffx, buffy = buffer
        bounds[0]-=buffx
        bounds[1]-=buffy
        bounds[2]+=buffx
        bounds[3]+=buffy
    bounds[1], bounds[2] = bounds[2], bounds[1]
    return BoundingBox(*bounds, 0.0, 9.223372036854776e+18)

def skip_none_collate(batch):
    """Collate function that filters out None samples."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return stack_samples(batch)

class RandomZeroMask:
    """
    Transform to replace random frames with zeros
    """
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, x):           # x: (T, N_patches, D) / (T, D)
        T = x.shape[0]
        zero_mask = torch.rand(T) > self.p
        # always keep at least min_keep frames
        if zero_mask.sum() == 0:
            zero_mask[np.random.randint(T)] = True
        shape = (T,) + (1,)*(len(x.shape)-1)
        return x * zero_mask.view(*shape)


class TileWindowDataset(Dataset):
    """
    Loads one .npy file of shape (T_total, N, N_patches, embed_dim) and
    exposes every (tile, time_window) pair as an individual sample.

    Each sample is:
        vec     — (T, N_patches, embed_dim)  input window
        target  — (N_patches, embed_dim)     frame immediately after the window
        tile_idx — int, for debugging / analysis
    """

    def __init__(self, filepath: str, T: Union[int, tuple], transform = None, p: float=0.2):
        # feat_data: (T_total, N, N_patches, embed_dim)
        
        self.data = torch.from_numpy(
            np.load(filepath, mmap_mode='r')
        )
        #self.data = self.data[:, ::10]
        #print(f"{os.path.basename(filepath)} : {self.data.shape}")
        
        self.T       = T
        self.T_total = self.data.shape[0]
        self.N       = self.data.shape[1]
        self.T_max = T[-1] if isinstance(T, tuple) else T
        self.transform = transform

        # build flat index: list of (tile_idx, t) pairs
        indices = []
        for n in range(self.N):
            selected_inds = []
            for t in range(self.T_max, self.T_total):           #self.T_total - self.T_max
                if np.random.random()<p:
                    selected_inds.append(t)
            if not selected_inds:
                selected_inds.append(np.random.randint(self.T_max, self.T_total))           #self.T_total - self.T_max
            indices.extend([(n, t) for t in selected_inds])

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        vec    = self.data[t : t + self.T_max, n]       # (T, N_patches, embed_dim)
        target = self.data[t + self.T_max, n]           # (N_patches, embed_dim)

        if self.transform is not None:
            vec = self.transform(vec)
        return vec, target

def run_epoch(classifier, loader, optimizer, loss_fn, train: bool, scheduler = None):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.train() if train else classifier.eval()
    total_loss, n_steps = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for vec, target in loader:
            # vec:    (B, T, N_patches, embed_dim)
            # target: (B, N_patches, embed_dim)
            vec    = vec.to(DEVICE)
            target = target.to(DEVICE).detach()     # stop-gradient on target
            predicted = classifier(vec)             # (B, N_patches, embed_dim)
            loss      = loss_fn(predicted, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            n_steps    += 1

    return total_loss / n_steps

def train_loop(classifier: nn.Module, feat_dir: str, out_path: str, T: Union[int, tuple], loss_fn: nn.Module, batch_size: int, N_epochs: int, lr: float, p_mask: float = 0.05, n_workers: int = 1):

    def collate_random_T(batch):
        vecs, targets = zip(*batch)
        if isinstance(T, tuple):
            rd_T = np.random.randint(T[0], T[1])
        else:
            rd_T = T
        vecs   = torch.stack([v[-rd_T:] for v in vecs])   # (B, T, N_patches, D)
        targets = torch.stack(list(targets))            # (B, N_patches, D)
        return vecs, targets

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_files = sorted(os.listdir(feat_dir))
    sep = int(np.floor(3*len(features_files)/4))
    print(sep)
    train_files = features_files[:sep]#[:1]
    val_files = features_files[sep:]#[:1]
    print("Train files : \n", train_files, "\nVal files : \n", val_files)

    transforms = RandomZeroMask(p=p_mask)
    
    optimizer = Adam(classifier.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    classifier.to(DEVICE)
    train_losses, val_losses = [], []
    for epoch in range(N_epochs):
        print(f"\nEpoch {epoch+1}/{N_epochs}")

        # Training
        train_loss = 0.0
        print("Train : ")
        for f in tqdm(train_files):
            dataset = TileWindowDataset(os.path.join(feat_dir, f), T=T, transform=transforms)
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn = collate_random_T,
            )
            train_loss += run_epoch(classifier, loader, optimizer, loss_fn, scheduler=scheduler, train=True)
        train_loss /= max(len(train_files), 1)
        train_losses.append(train_loss)
        # Val
        val_loss = 0.0
        print("\nValidation : ")
        for f in tqdm(val_files):
            dataset = TileWindowDataset(os.path.join(feat_dir, f), T=T, transform=transforms)
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn = collate_random_T,
            )
            val_loss += run_epoch(classifier, loader, optimizer, loss_fn, scheduler=scheduler, train=False)
        val_loss /= max(len(val_files), 1)

        print(f"  train loss: {train_loss:.6f}  |  val loss: {val_loss:.6f}")

        val_losses.append(val_loss)
        if val_loss == np.min(val_losses):
            torch.save(classifier.state_dict, out_path)

    X = np.array([i+1 for i in range(N_epochs)])
    plt.plot(X, train_losses)
    plt.plot(X, val_losses)
    plt.show()

    return train_losses, val_losses