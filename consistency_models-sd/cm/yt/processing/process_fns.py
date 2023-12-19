import base64
import io
import sys
from typing import Any, List, Union

import cv2
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf, ListConfig
import yt
import yt.yson as yson
from cm.yt.encoding.process_fns import NumpyEncoding
from cm.yt.processing.frame_samplers import UniformFrameSampler
from cm.yt.utils import instantiate_from_config
from numpy.typing import NDArray

try:
    import decord
except ImportError:
    decord = None


def to_tensor(
    val: Union[float, int, NDArray], dtype: Union[str, torch.dtype] = None
) -> torch.Tensor:
    _STR_TO_DTYPE = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if isinstance(dtype, str):
        dtype = _STR_TO_DTYPE[dtype]
    return torch.as_tensor(val, dtype=dtype)


def normalize_image(val):
    return val / 127.5 - 1.0


def string_decode(string: str):
    if string is None:
        string=b""
    return string.decode("utf-8")


def identity(data: bytes) -> bytes:
    return data


def stack(dtype: np.dtype = None, *args):
    return np.array(args, dtype=dtype)


def decode_image_from_bytes(
    image_bytes,
    dtype: np.dtype = np.uint8,
    use_base64: bool = False,
    transpose_hwc2chw: bool = True,
) -> np.array:
    image_bytes = yson.get_bytes(image_bytes)
    if use_base64:
        image_bytes = base64.b64decode(image_bytes)

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=dtype), cv2.IMREAD_ANYCOLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if transpose_hwc2chw:
        image = np.transpose(image, [2, 0, 1])

    return image


def decode_mask_from_bytes(image_bytes, dtype=np.uint8):
    image_bytes = yson.get_bytes(image_bytes)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=dtype), cv2.IMREAD_GRAYSCALE)

    if len(image.shape) == 2:
        image = image[:, :, None]

    image = np.transpose(image, [2, 0, 1])
    image = image.astype(np.float32) / 255.0

    return image


def crop_image(
        img: np.ndarray,
        mask: np.ndarray,
) -> List[torch.FloatTensor]:
    h, w, _ = img.shape
    crop = min(h, w)
    slice_h, slice_w = slice((h - crop) // 2, (h + crop) // 2), slice(
        (w - crop) // 2, (w + crop) // 2
    )
    img = img[slice_h, slice_w, :]
    mask = mask[slice_h, slice_w]
    return img, mask


def pad_image(
        img: np.ndarray,
        mask: np.ndarray,
) -> List[torch.FloatTensor]:
    h, w, _ = img.shape
    ext = max(h, w)
    pad_h, pad_w = ((ext - h) // 2,), ((ext - w) // 2,)
    img = np.pad(img, (pad_h * 2, pad_w * 2, (0, 0)), "edge")
    mask = np.pad(mask, (pad_h * 2, pad_w * 2), "constant", constant_values=((0, 0), (0, 0)))
    return img, mask


def decode_nparray_from_bytes(arr_bytes, dtype: np.dtype = np.float32):
    arr_bytes = yson.get_bytes(arr_bytes)
    arr = np.frombuffer(arr_bytes, dtype=dtype)
    return arr


def decode_latent_nparray_from_bytes(image_bytes, dtype: np.dtype = np.float32):
    arr_bytes = yt.yson.get_bytes(image_bytes)
    arr = np.frombuffer(arr_bytes, dtype=dtype)
    arr = np.reshape(arr, (4, 64, 64))
    return to_tensor(arr, dtype=torch.float32)


def uniform_video_sampler(
    video_bytes: Any,
    num_frames: int,
    frame_step: int,
    side_size: int = 64,
    error_mode: str = "sample_first",
) -> torch.Tensor:
    if decord is None:
        raise ImportError("Module decord is not installed.")

    byte_stream = io.BytesIO(video_bytes)
    vr = decord.VideoReader(byte_stream)

    total_num_frames = len(vr)
    sampler = UniformFrameSampler(num_frames=num_frames, frame_step=frame_step, error_mode=error_mode)

    total_num_frames = len(vr)
    frame_ids  = sampler(total_num_frames)

    decord.bridge.set_bridge("torch")

    frames = vr.get_batch(frame_ids).permute(0, 3, 1, 2)
    del vr
    frames = transforms.CenterCrop(side_size)(frames)
    frames = frames.to(dtype=torch.float32)

    frames.div_(127.5).sub_(1)

    return frames


def decode_image_pipeline(image_bytes):
    # TODO:  REPLACE WITH THE AUGMENTOR CLASS
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=False)
    image = np.transpose(image, [2, 0, 1])
    image = to_tensor(image, dtype=torch.float32)
    return normalize_image(image)


def decode_image_pipeline_with_augmentations(
    image_bytes, aug_pipeline_config=None
):
    image = decode_image_from_bytes(image_bytes, transpose_hwc2chw=False)
    if aug_pipeline_config is not None:
        aug_pipeline = instantiate_from_config(
            aug_pipeline_config
        )
        image = aug_pipeline(image)
    image = np.transpose(image, [2, 0, 1])
    image = to_tensor(image, dtype=torch.float32)
    return normalize_image(image)


def decode_embeddings_pipeline(arr_bytes, text_embedding_dim: int = 4096, encoding: Union[NumpyEncoding, str] = NumpyEncoding.BYTES):
    if isinstance(encoding, str):
        encoding = NumpyEncoding[encoding]

    if encoding == NumpyEncoding.BYTES:
        decoded = decode_nparray_from_bytes(arr_bytes, dtype=np.float16)
        decoded = decoded.reshape(-1, text_embedding_dim)
    elif encoding == NumpyEncoding.BUFFER:
        arr_bytes = yson.get_bytes(arr_bytes)
        decoded = np.load(io.BytesIO(arr_bytes))
    else:
        raise NotImplementedError()

    text_mask = np.ones(decoded.shape[0], dtype=bool)
    return to_tensor(decoded, dtype=torch.float32), to_tensor(
        text_mask, dtype=torch.float32
    )


def decode_numpy_pipeline(arr_bytes, dims: List[int] = [3, 64, 64]):
    decoded = decode_nparray_from_bytes(arr_bytes, dtype=np.float16)
    reshaped = decoded.reshape(*dims)
    return to_tensor(reshaped, dtype=torch.float32)


def decode_pos_mask_pipeline(arr_bytes):
    decoded = decode_nparray_from_bytes(arr_bytes, dtype=bool).astype(np.float32)
    return to_tensor(decoded, dtype=torch.float32)


def prepare_continuous_pipeline(*args):
    stacked = stack(*args, dtype=np.float32)
    return to_tensor(stacked, dtype=torch.float32)


def prepare_discrete_pipeline(val):
    val = np.array([val], dtype=np.int32)
    return to_tensor(val, dtype=torch.int32)


def decode_continuous_from_bytes(arr_bytes):
    decoded = decode_nparray_from_bytes(arr_bytes)
    return to_tensor(decoded, dtype=torch.float32)


# dtypes cannot be specified from config
def _get_dummy_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    dtype: torch.dtype,
    squeeze: bool = False,
) -> torch.Tensor:

    if isinstance(size, ListConfig):
        size = OmegaConf.to_container(size)

    tensor = torch.full(size, fill_value, dtype=dtype)
    if squeeze:
        tensor = tensor.squeeze_()

    return tensor


def get_dummy_float32_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    squeeze: bool = False,
) -> torch.Tensor:
    return _get_dummy_tensor(size=size, fill_value=fill_value, dtype=torch.float32, squeeze=squeeze)


def get_dummy_bool_tensor(
    size: Union[ListConfig, List[int]],
    fill_value: Any,
    squeeze: bool = False,
) -> torch.Tensor:
    return _get_dummy_tensor(size=size, fill_value=fill_value, dtype=torch.bool, squeeze=squeeze)


def get_empty_string(*args, **kwargs) -> str:
    return ""
