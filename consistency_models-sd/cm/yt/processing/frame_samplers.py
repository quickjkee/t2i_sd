from enum import Enum
import random
from typing import List, Union


# StrEnum cannot be used because of old python
class ErrorMode(str, Enum):
    SAMPLE_FIRST = "sample_first"
    RAISE_ERROR = "raise_error"


class UniformFrameSampler:
    def __init__(
        self,
        num_frames: int,
        frame_step: int,
        error_mode: Union[ErrorMode, str] = ErrorMode.SAMPLE_FIRST
    ) -> None:
        self._num_frames = num_frames
        self._step = frame_step
        self._length = num_frames * frame_step
        self._error_mode = ErrorMode(error_mode)

    def __call__(self, total_num_frames: int) -> List[int]:
        max_offset = total_num_frames - self._length

        if max_offset >= 0:
            start = random.randint(0, max_offset)
            return list(range(start, start + self._length, self._step))

        if self._error_mode is ErrorMode.SAMPLE_FIRST:
            return self._num_frames * [0]

        raise RuntimeError("Video too short!")

