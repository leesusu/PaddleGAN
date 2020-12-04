#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import cv2
import traceback
import numpy as np
from subprocess import getstatusoutput

import paddle
from ppgan.models.generators import Wav2Lip
from ppgan.utils.download import get_path_from_url
from ppgan.utils.logger import get_logger
from ppgan.utils.audio import load_wav, melspectrogram
from ppgan.faceutils.face_detection import FaceAlignment, LandmarksType

from .base_predictor import BasePredictor

WAV2LIP_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/wav2lip.pdparams'


class Wav2LipPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 static=False,
                 fps=25,
                 resize_factor=1,
                 rotate=False,
                 smooth=True,
                 crop=[0, -1, 0, -1],
                 box=[-1, -1, -1, -1],
                 pads=[0, 10, 0, 0],
                 wav2lip_batch_size=128,
                 face_det_batch_size=16):
        self.logger = get_logger()

        self.output = os.path.join(output, 'Wav2Lip')
        self.tmp_dir = os.path.join(self.output, 'tmp')

        if weight_path is None:
            weight_path = get_path_from_url(WAV2LIP_WEIGHT_URL)

        self.model = Wav2Lip()
        state_dict = paddle.load(weight_path)
        self.model.load_dict(state_dict)
        self.model.eval()
        self.logger.info('Model loaded.')

        self.static = static
        self.fps = fps
        self.resize_factor = resize_factor
        self.rotate = rotate
        self.smooth = smooth
        self.crop = crop
        self.box = box
        self.pads = pads
        self.wav2lip_batch_size = wav2lip_batch_size
        self.face_det_batch_size = face_det_batch_size

        self.mel_step_size = 16
        self.face_img_size = 96
        self.T = 5

    def get_smoothened_boxes(self, boxes):
        for i in range(len(boxes)):
            if i + self.T > len(boxes):
                window = boxes[len(boxes) - self.T:]
            else:
                window = boxes[i:i + self.T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = FaceAlignment(LandmarksType._2D, flip_input=False)

        batch_size = self.face_det_batch_size

        predictions = []
        while True:
            try:
                for i in range(0, len(images), batch_size):
                    predictions.extend(
                        detector.get_detections_for_batch(
                            np.array(images[i:i + batch_size])))
            except:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument'
                    )
                batch_size //= 2
                self.logger.info(
                    'Recovering from OOM error, new batch size: %d' %
                    batch_size)
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                # check this frame where the face was not detected.
                tmp_faulty_frame_path = os.path.join(self.tmp_dir,
                                                     'faulty_frame.jpg')
                cv2.imwrite(tmp_faulty_frame_path, image)
                raise ValueError(
                    'Face not detected! Ensure the video contains a face in all the frames.'
                )

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if self.smooth:
            boxes = self.get_smoothened_boxes(boxes)

        results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                   for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if self.static:
                face_det_results = self.face_detect([frames[0]])
            else:
                # BGR2RGB for CNN face detection
                face_det_results = self.face_detect(frames)
        else:
            self.logger.info(
                'Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                                for f in frames]

        def _parse_img_mel(img_batch, mel_batch):
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.face_img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.expand_dims(mel_batch, axis=-1)

            img_batch = np.transpose(img_batch, (0, 3, 1, 2))
            mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))

            img_batch = paddle.to_tensor(img_batch, dtype='float32')
            mel_batch = paddle.to_tensor(mel_batch, dtype='float32')

            return img_batch, mel_batch

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.face_img_size, self.face_img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = _parse_img_mel(img_batch, mel_batch)
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = _parse_img_mel(img_batch, mel_batch)
            yield img_batch, mel_batch, frame_batch, coords_batch

    def _run(self, face_path, audio_path):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        if not os.path.isfile(face_path):
            raise ValueError(
                'face_path argument must be a valid path to video/image file.')

        face_ext_name = face_path.split('.')[-1]
        if face_ext_name in ['jpg', 'jpeg', 'png']:
            full_frames = [cv2.imread(face_path)]
        elif face_ext_name == 'mp4':
            video_stream = cv2.VideoCapture(face_path)
            self.fps = video_stream.get(cv2.CAP_PROP_FPS)

            self.logger.info('Reading video frames...')

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break

                if self.resize_factor > 1:
                    frame = cv2.resize(frame,
                                       (frame.shape[1] // self.resize_factor,
                                        frame.shape[0] // self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)
        else:
            raise ValueError(
                'face_path argument must be a valid path with .jpg/.jpeg/.png/.mp4 extension name.'
            )

        frame_h, frame_w = full_frames[0].shape[:-1]

        self.logger.info('Number of frames available for inference: %d' %
                         len(full_frames))

        if not audio_path.endswith('.wav'):
            self.logger.info('Extracting raw audio...')

            tmp_audio_path = os.path.join(self.tmp_dir, 'audio.wav')
            cmd = 'ffmpeg -y -i %s -strict -2 %s' % (audio_path, tmp_audio_path)

            cmd_status, cmd_output = getstatusoutput(cmd)
            if cmd_status != 0:
                raise RuntimeError(cmd_output)

            audio_path = tmp_audio_path

        wav = load_wav(audio_path, 16000)
        mel = melspectrogram(wav)

        if True in np.isnan(mel):
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again.'
            )

        mel_chunks = []
        mel_idx_multiplier = 80.0 / self.fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + self.mel_step_size])
            i += 1

        self.logger.info('Length of mel chunks: %d' % len(mel_chunks))

        full_frames = full_frames[:len(mel_chunks)]
        gen = self.datagen(full_frames.copy(), mel_chunks)

        tmp_result_path = os.path.join(self.tmp_dir, 'result.avi')
        out = cv2.VideoWriter(tmp_result_path, cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps, (frame_w, frame_h))

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            with paddle.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        result_path = os.path.join(self.output, 'result.avi')
        cmd = 'ffmpeg -y -i %s -i %s -strict -2 -q:v 1 %s' % (
            audio_path, tmp_result_path, result_path)
        cmd_status, cmd_output = getstatusoutput(cmd)
        if cmd_status != 0:
            raise RuntimeError(cmd_output)

    def run(self, face_path, audio_path):
        try:
            self._run(face_path, audio_path)
        except Exception as e:
            self.logger.error(traceback.format_exc())
