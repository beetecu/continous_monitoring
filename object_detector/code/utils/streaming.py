
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

import cv2 as cv


class MultiStreamerCapture:
    def __init__(self, logger, sources):
        assert sources
        self.logger = logger
        self.captures = []

        try:
            sources = [int(src) for src in sources]
            mode = 'cam'
        except ValueError:
            mode = 'video'

        if mode == 'cam':
            for id in sources:
                self.logger.info('INFO', 'Connection  cam {}'.format(id))
                cap = cv.VideoCapture(id)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv.CAP_PROP_FPS, 30)
                cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
                assert cap.isOpened()
                self.captures.append(cap)
        else:
            for stream_path in sources:
                self.logger.info('INFO', 'Opening file {}'.format(stream_path))
                input_stream = "souphttpsrc location=" + stream_path + " ! hlsdemux ! decodebin ! videoconvert ! videoscale ! appsink max-buffers=1 drop=true"
                cap = cv.VideoCapture(input_stream, cv.CAP_GSTREAMER)
                assert cap.isOpened()
                self.captures.append(cap)

    def get_frames(self):
        frames = []
        for capture in self.captures:
            has_frame, frame = capture.read()
            if has_frame:
                frames.append(frame)

        return len(frames) == len(self.captures), frames

    def get_num_sources(self):
        return len(self.captures)
