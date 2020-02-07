"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from utils.ie_tools import load_ie_model


class Detector:
    """Wrapper class for detector"""

    def __init__(self, logger, model_path, conf=.6, device='CPU', ext_path='', max_num_frames=1 ):
        self.net = load_ie_model(logger, model_path, device, None, ext_path, num_reqs=max_num_frames)
        self.confidence = conf
        self.max_num_frames = max_num_frames
        self.logger = logger

    def get_detections(self, frames):
        """Returns all detections on frames"""
        assert len(frames) <= self.max_num_frames

        all_detections = []
        for i in range(len(frames)):
            self.net.forward_async(frames[i])
        outputs = self.net.grab_all_async()

        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, frames[i].shape)
            all_detections.append(detections)

        return all_detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                class_id = int(max(detection[1], 0))

                detections.append(((left, top, right, bottom), confidence, class_id))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections
