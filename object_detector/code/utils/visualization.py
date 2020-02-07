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
import numpy as np
from utils.misc import COLOR_PALETTE


def draw_detections(frame, detections,  labels_map):
    """Draws detections and labels"""
    for i, obj in enumerate(detections):
        left, top, right, bottom = detections[i][0]
        class_id = detections[i][2]
        label_c = '%.2f' % detections[i][1]
        # Get the label for the class name and its confidence
        label_objects = labels_map[class_id] if labels_map else str(class_id)
        label = label_objects + " | " + label_c
        box_color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
        cv.line(frame, (left, top), (left, bottom), box_color, 3)
        cv.line(frame, (right, top), (right, bottom), box_color, 3)
        overlay = frame.copy()
        cv.rectangle(overlay, (left, top), (right, bottom), box_color, -1)

        opacity = 0.2
        cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def visualize_detections(frames, all_objects, labels_map, fps=''):
    assert len(frames) == len(all_objects)
    vis = None
    for frame, objects in zip(frames, all_objects):
        new_frame = draw_detections(frame, objects, labels_map)
        if vis is not None:
            vis = np.vstack([vis, new_frame])
        else:
            vis = new_frame

    n_cams = len(frames)
    vis = cv.resize(vis, (vis.shape[1] // n_cams, vis.shape[0] // n_cams))

    label_size, base_line = cv.getTextSize(str(fps),
                                           cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv.putText(vis, str(fps), (base_line*2, base_line*3),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return vis
