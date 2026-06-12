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

from torch.utils.data import DataLoader, Sampler, Dataset
from torch.optim import Adam
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



class TileWindowDataset(Dataset):
    """
    Loads one .npy file of shape (T_total, N, N_patches, embed_dim) and
    exposes every (tile, time_window) pair as an individual sample.

    Each sample is:
        vec     — (T, N_patches, embed_dim)  input window
        target  — (N_patches, embed_dim)     frame immediately after the window
        tile_idx — int, for debugging / analysis
    """

    def __init__(self, filepath: str, T: int):
        # feat_data: (T_total, N, N_patches, embed_dim)
        self.data = torch.from_numpy(
            np.load(filepath),
            dtype=torch.float32,
        )
        print(f"{filepath} : {self.data.shape}")
        self.T       = T
        self.T_total = self.data.shape[0]
        self.N       = self.data.shape[1]

        # build flat index: list of (tile_idx, t) pairs
        self.indices = [
            (n, t)
            for n in range(self.N)
            for t in range(self.T_total - self.T)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        vec    = self.data[t : t + self.T, n]       # (T, N_patches, embed_dim)
        target = self.data[t + self.T, n]           # (N_patches, embed_dim)
        return vec, target


def run_epoch(classifier, loader, optimizer, loss_fn, train: bool):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.train() if train else classifier.eval()
    total_loss, n_steps = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for vec, target in tqdm(loader):
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

            total_loss += loss.item()
            n_steps    += 1

    return total_loss / n_steps


def train_loop(classifier: nn.Module, feat_dir: str, out_path: str, T: int, loss_fn: nn.Module, batch_size: int, N_epochs: int, lr: float, n_workers: int = 1):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_files = os.listdir(feat_dir)
    sep = int(np.ceil(3*len(features_files)/4))
    train_files = features_files[:sep]
    val_files = features_files[sep:]
    print("Train files : \n", train_files, "\nVal files : \n", val_files)
    
    optimizer = Adam(classifier.parameters(), lr=lr)
    classifier.to(DEVICE)

    for epoch in range(N_epochs):
        print(f"\nEpoch {epoch+1}/{N_epochs}")

        # Training
        train_loss = 0.0
        for f in tqdm(train_files, desc="train files"):
            dataset = TileWindowDataset(os.path.join(feat_dir, f), T=T)
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_workers,
                pin_memory=True,
            )
            train_loss += run_epoch(classifier, loader, optimizer, loss_fn, train=True)
        train_loss /= max(len(train_files), 1)

        # Val
        val_loss = 0.0
        for f in tqdm(val_files, desc="val files"):
            dataset = TileWindowDataset(os.path.join(feat_dir, f), T=T)
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=True,
            )
            val_loss += run_epoch(classifier, loader, optimizer, loss_fn, train=False)
        val_loss /= max(len(val_files), 1)

        print(f"  train loss: {train_loss:.6f}  |  val loss: {val_loss:.6f}")

        # -- checkpoint ------------------------------------------------------
        torch.save({
            "epoch":      epoch + 1,
            "model":      classifier.state_dict(),
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }, out_path)