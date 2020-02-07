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

import argparse
import os
import queue
from threading import Thread
import json
import logging as log
import sys
import time

import cv2 as cv

from utils.log_sender import logger
from utils.network_wrappers import Detector, VectorCNN
from mc_tracker.mct import MultiCameraTracker
from utils.misc import read_py_config
from utils.streaming import MultiStreamerCapture
from utils.visualization import visualize_multicam_detections

### Devo Sender
from devo.sender import Sender, SenderConfigSSL, SenderConfigTCP

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

DEFAULT_SERVER = "eu.elb.relay.logtrust.net"
DEFAULT_PORT = 443
DEFAULT_TABLE_NAME = 'my.app.tracker_box.info'
DEFAULT_STAT_TABLE_NAME = 'my.app.tracker.stats'
DEFAULT_SOURCE = "unknown"

KEY = "CERTS/key.key"
CERT = "CERTS/cert.crt"
CHAIN = "CERTS/chain.crt"


class FramesThreadBody:
    def __init__(self, capture, timer, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length
        self.timer = timer

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(self.timer)
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def run(params, capture, detector, reid):

    config = {}
    if len(params.config):
        config = read_py_config(params.config)

    if params.debug:
        win_name = 'Multi camera tracking'
    else:
        engine_config = SenderConfigSSL(address=(params.devo_server, params.devo_port), key=KEY, cert=CERT, chain=CHAIN)
        sender = Sender(engine_config)

    # arreglar que el tamaño sea de acuerdo a lo que entra
    if params.broadcast:
        GST_PIPE = "appsrc is-live=1 \
                ! videoconvert \
                    ! video/x-raw, width=1920, height=1080, framerate=1/1 \
                    ! queue \
                    ! x264enc bitrate=4500 byte-stream=false key-int-max=60 bframes=0 aud=true tune=zerolatency \
                    ! video/x-h264,profile=main \
                    ! flvmux streamable=true name=mux \
                    ! rtmpsink location={0} audiotestsrc \
                    ! voaacenc bitrate=128000 \
                    ! mux.".format(params.broadcast)

        out_send = cv.VideoWriter(GST_PIPE, cv.CAP_GSTREAMER, 0, 1, (1920, 1080), True)

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, **config)

    thread_body = FramesThreadBody(capture, params.processing_timer, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    if len(params.output_video):
        video_output_size = (1920 // capture.get_num_sources(), 1080)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video = cv.VideoWriter(params.output_video,
                                      fourcc, 24.0,
                                      video_output_size)
    else:
        output_video = None

    while cv.waitKey(1) != 27 and thread_body.process:
        start = time.time()
        try:
            frames = thread_body.frames_queue.get_nowait()
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.get_detections(frames)
        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        if params.debug:
            fps = round(1 / (time.time() - start), 1)
            vis = visualize_multicam_detections(frames, tracked_objects, fps)
            cv.imshow(win_name, vis)
            if output_video:
                output_video.write(cv.resize(vis, video_output_size))
            if params.broadcast:
                out_send.write(vis)
        else:
            # engine_config = SenderConfigTCP(address=('localhost', 1514))
            logger(params.tablename, sender, frames, tracked_objects, params.source)

        end = time.time()
        if (params.processing_timer - (end - start)) > 0:
            time.sleep(params.processing_timer - (end - start))
            if params.broadcast:
                out_send.write(visualize_detections(frames, detections, labels_map, fps))


    thread_body.process = False
    frames_thread.join()


    if len(params.history_file):
        history = tracker.get_all_tracks_history()
        with open(params.history_file, 'w') as outfile:
            json.dump(history, outfile)




def main():
    """Prepares data for the person recognition demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking live demo script')
    parser.add_argument('-i', type=str, nargs='+', help='List of cameras url address', required=True)

    parser.add_argument('-m', '--m_detector', type=str,
                        help='Path to the person detection model',
                        default='models/detector/person-detection-retail-0013.xml')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the person detection model')
    parser.add_argument('--m_reid', type=str,
                        help='Path to the person reidentification model',
                        default='models/detector/person-reidentification-retail-0079.xml')

    parser.add_argument('--output_video', type=str, default='', required=False)
    parser.add_argument('--config', type=str, default='', required=False)
    parser.add_argument('--history_file', type=str, default='', required=False)

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str,
                      default='/root/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so')

    parser.add_argument('-p', "--processing_timer", help='Processing time step (in seconds)', default=0.1, type=float)
    parser.add_argument('-t', '--tablename', help='Table Name.', default=DEFAULT_TABLE_NAME, type=str)
    parser.add_argument('-ds', '--devo_server', help='Devo Server.', default=DEFAULT_SERVER, type=str)
    parser.add_argument('-dp', '--devo_port', help='Devo Port.', default=DEFAULT_PORT, type=str)
    
    parser.add_argument('-b', "--broadcast", help='Streaming broadcast address.', type=str)
    parser.add_argument('-db', "--debug", help='Debug Mode: No logs are sent and the images process result is show in a windows ', action="store_true")
    parser.add_argument('-s', "--source", help='Stream source identifier.', type=str, default=DEFAULT_SOURCE)

    args = parser.parse_args()

    capture = MultiStreamerCapture(args.i)

    person_detector = Detector(args.m_detector,
                               args.t_detector,
                               args.device, args.cpu_extension,
                               capture.get_num_sources())
    if args.m_reid:
        person_recognizer = VectorCNN(args.m_reid, args.device)
    else:
        person_recognizer = None
    run(args, capture, person_detector, person_recognizer)
    log.info('Demo finished successfully')

if __name__ == '__main__':
    main()
