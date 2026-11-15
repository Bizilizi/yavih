# Copied from CAV-MAE (Gong et al., ICLR 2023), the frame extraction the AVROBUSTBENCH README names as its
# preprocessing ("To extract the frames and audio from the videos ... please refer to this repo"):
#   repository https://github.com/YuanGongND/cav-mae
#   commit     68fe8c2a3917dc2926e41f796bfdcb331a64b42c
#   path       src/preprocess/extract_video_frame.py (`preprocess`, `extract_frame`)
#   licence    BSD 2-Clause (Copyright (c) 2023, Yuan Gong; the notice is kept at the end of this file)
# Changes:
#   1. extract_frame returns the 10 frames in memory instead of writing target_fold/frame_<i>/<video_id>.jpg: each
#      frame goes through torchvision's save_image into an in-memory JPEG (format "JPEG", PIL's default quality, as
#      save_image(..., '<id>.jpg') writes it) and is re-opened with PIL, which is what AVROBUSTBENCH's dataset.py
#      reads (Image.open(<frame jpg>));
#   2. the progress print is dropped, and a frame that cv2 cannot read raises FrameReadError (upstream: the cvtColor
#      on None raises and the caller's bare `except` prints "error with" and skips the video);
#   3. the video_id parsing and the os.makedirs of the frame folders are dropped (nothing is written).
import io

import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])


class FrameReadError(RuntimeError):
    """cv2 returned no frame at an index the recipe asks for (yavih change 2)."""


def extract_frame(input_video_path, extract_frame_num=10):
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
    frames = []
    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        if frame is None:  # yavih change 2
            raise FrameReadError(f"{input_video_path}: no frame at index {frame_idx - 1} (frame {i})")
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        # yavih change 1: save_image to memory (upstream: target_fold + '/frame_{:d}/'.format(i) + video_id + '.jpg')
        buf = io.BytesIO()
        save_image(image_tensor, buf, format="JPEG")
        buf.seek(0)
        frames.append(Image.open(buf))
    vidcap.release()
    return frames


# ---------------------------------------------------------------------------------------------------------------------
# Upstream licence (https://github.com/YuanGongND/cav-mae/blob/68fe8c2a3917dc2926e41f796bfdcb331a64b42c/LICENSE):
#
# BSD 2-Clause License
#
# Copyright (c) 2023, Yuan Gong
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
