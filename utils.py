import os
import importlib
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ning.data.utils import custom_collate
import cv2
# from torchvision.utils import save_image
import torch
import math
import pathlib
import warnings

from typing import Any, BinaryIO, List, Optional, Tuple, Union
from types import FunctionType

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig():
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)



def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        range (tuple. optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``value_range``
                instead.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs.keys():
        warnings.warn(
            "The parameter 'range' is deprecated since 0.12 and will be removed in 0.14. "
            "Please use 'value_range' instead."
        )
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.add_(1.0).mul(127.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def save_img(log, save_config):

    inputs = log['inputs'].detach()
    reconstructions = log['reconstructions'].detach()

    path = save_config['imgdir']+'/inputs_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(inputs, path)

    path = save_config['imgdir']+'/reconstructions_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(reconstructions, path)

    # path = save_config['imgdir']+'/inputs_gs-{:06}_e-{:06}_b-{:06}.png'.format(
    #     save_config['split'], save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    # save_image(inputs, path)

    # path = save_config['imgdir']+'/reconstructions_gs-{:06}_e-{:06}_b-{:06}.png'.format(
    #     save_config['split'], save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    # save_image(reconstructions, path)


def save_ddpm_img(log, save_config):

    inputs = log['inputs'].detach()
    diffusion_row = log['diffusion_row'].detach()
    samples = log['samples'].detach()
    denoise_row = log['denoise_row'].detach()

    path = save_config['imgdir']+'/inputs_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(inputs, path)

    path = save_config['imgdir']+'/diffusion_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(diffusion_row, path)

    path = save_config['imgdir']+'/samples_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(samples, path)

    path = save_config['imgdir']+'/denoise_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(denoise_row, path)


def save_ldm_img(log, save_config):

    inputs = log['inputs'].detach()
    reconstruction = log['reconstruction'].detach()
    conditioning = log['conditioning'].detach()
    diffusion_row = log['diffusion_row'].detach()
    # denoise_row = log['denoise_row'].detach()
    samples = log['samples'].detach()
    samples_x0_quantized = log['samples_x0_quantized'].detach()

    path = save_config['imgdir']+'/inputs_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(inputs, path)

    path = save_config['imgdir']+'/reconstruction_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(reconstruction, path)

    path = save_config['imgdir']+'/conditioning_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(conditioning, path)

    path = save_config['imgdir']+'/diffusion_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(diffusion_row, path)

    # path = save_config['imgdir']+'/denoise_row_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
    #     save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    # save_image(denoise_row, path)

    path = save_config['imgdir']+'/samples_row_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(samples, path)

    path = save_config['imgdir']+'/samples_x0_quantized_row_row_gs-{:06}_e-{:06}_b-{:06}.png'.format(
        save_config['global_step'], save_config['epoch'], save_config['batch_index'])
    save_image(samples_x0_quantized, path)


