"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissi#ons and
 limitations under the License.
"""

import argparse
import queue
from threading import Thread
import time

import cv2 as cv

from log_sender import Log

from utils.network_wrappers import Detector, VectorCNN
from mc_tracker.mct import MultiCameraTracker
from utils.misc import read_py_config
from utils.streaming import MultiStreamerCapture
from utils.visualization import visualize_multicam_detections

from watchdog_timer import WDT
import os
from multiprocessing import Process
from functools import partial
import signal
import logging as log
import sys

log.basicConfig(stream=sys.stdout, level=log.DEBUG)

DEFAULT_SERVER = "eu.elb.relay.logtrust.net"
DEFAULT_PORT = 443
DEFAULT_TABLE_NAME = 'my.app.person_tracker.tracking'
DEFAULT_STAT_TABLE_NAME = 'my.app.person_tracker.stats'
DEFAULT_INFO_TABLE_NAME = 'my.app.person_tracker.info'
DEFAULT_SOURCE = "unknown"

def isObjectsInLabels(objects, labels):
    return all(elem in labels for elem in objects)

def assert_exit(condition, err_message):
    try:
        assert condition
    except AssertionError:
        sys.exit(err_message)



class FramesThreadBody:
    def __init__(self, logger, capture, timer, pid, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length
        self.timer = timer
        self.pid = pid
        self.logger = logger
    
    def restart(self):
        try:
            import signal
            self.logger.info("ERROR", "error in capture -> RESTARTING -> stoping %s..." % (self.pid))
            os.kill(self.pid, signal.SIGUSR1)
        except Exception as e:
            self.logger.info("ERROR", "exception: restart: %s" % (e))
            pass

    def __call__(self):
        watchdog = WDT(self.logger, check_interval_sec=30, trigger_delta_sec=120, callback=self.restart)
        while True:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(self.timer)
            has_frames, frames = self.capture.get_frames()
            watchdog.update()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)

def run(params, pid, logger):

    capture = MultiStreamerCapture(logger, params.i)
    thread_body = FramesThreadBody(logger, capture, params.processing_timer, pid,
                                   max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    person_detector = Detector(logger,
                               params.m_detector,
                               params.t_detector,
                               params.device, params.cpu_extension,
                               capture.get_num_sources())

    if params.m_reid:
        person_recognizer = VectorCNN(logger, params.m_reid, params.device)
    else:
        person_recognizer = None

    config = {}
    if len(params.config):
        config = read_py_config(params.config)

    tracker = MultiCameraTracker(capture.get_num_sources(), person_recognizer, **config)

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


    if len(params.output_video):
        video_output_size = (1920 // capture.get_num_sources(), 1080)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video = cv.VideoWriter(params.output_video,
                                      fourcc, 24.0,
                                      video_output_size)
    else:
        output_video = None

    stat = {}

    while cv.waitKey(1) != 27:

        stat['start'] = time.time()
        try:
            stat['cap.start'] = time.time()
            frames = thread_body.frames_queue.get_nowait()
            stat['cap.end'] = time.time()
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        stat['inference.start'] = time.time()
        detections = person_detector.get_detections(frames)
        all_detections = person_detector.get_detections(frames)
        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        stat['inference.end'] = time.time()

        if params.debug:
            fps = round(1 / (time.time() - stat['start']), 1)
            vis = visualize_multicam_detections(frames, tracked_objects, fps)
            cv.imshow("Object detection", vis)
            if output_video:
                output_video.write(cv.resize(vis, video_output_size))
            if params.broadcast:
                out_send.write(vis)

            if params.broadcast:
                out_send.write(visualize_multicam_detections(frames, tracked_objects, fps))

        stat['end'] = time.time()
        if params.sendlogs:
            logger.detections(frames, tracked_objects)
            logger.stats(stat)
        dif = params.processing_timer - (time.time() - stat['start'])
        if (dif) > 0:
            time.sleep(dif)


    thread_body.process = False
    frames_thread.join()

def signal_handler(process, logger):
    logger.info("INFO", "SIGNAL received. Terminating process....")
    process.terminate()

def getArgs():
    parser = argparse.ArgumentParser(description='Person Tracker live demo script')
    parser.add_argument('-i', type=str, nargs='+', help='Input sources (indexes '
                                                        'of cameras or paths to video files)', required=True)
    parser.add_argument('-m', '--m_detector', type=str,
                        help='Path to the person detection model',
                        default='models/detector/person-detection-retail-0013.xml')
    parser.add_argument('--t_detector', type=float, default=0.6, help='Threshold for the person detection model')
    parser.add_argument('--m_reid', type=str,
                        help='Path to the person reidentification model',
                        default='models/detector/person-reidentification-retail-0079.xml')
    parser.add_argument('--output_video', type=str, default='', required=False)
    parser.add_argument('--config', type=str, default='', required=False)
    parser.add_argument('--history_file', type=str, default='', required=False)
    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument("-l", "--cpu_extension",
                        help="Optional. Required for CPU custom layers. "
                             "Absolute path to a shared library with the kernels implementations inside the docker",
                        type=str,
                        default='/root/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so')
    parser.add_argument('-p', "--processing_timer", help='Processing time step (in seconds)', default=0.1, type=float)
    parser.add_argument('-dt', '--tdetect', help='Object Detected Table Name.', default=DEFAULT_TABLE_NAME, type=str)
    parser.add_argument('-st', '--tstat', help='Stats Table Name.', default=DEFAULT_STAT_TABLE_NAME, type=str)
    parser.add_argument('-it', '--tinfo', help='Info Table Name.', default=DEFAULT_INFO_TABLE_NAME, type=str)
    parser.add_argument('-ds', '--devo_server', help='Devo Server.', default=DEFAULT_SERVER, type=str)
    parser.add_argument('-dp', '--devo_port', help='Devo Port.', default=DEFAULT_PORT, type=str)
    parser.add_argument('-b', "--broadcast", help='Streaming broadcast address.', type=str)
    parser.add_argument('-db', "--debug", help='Show a window with the image detection results', action="store_true")
    parser.add_argument('-log', "--sendlogs", help='Send Logs to Devo Platform', action="store_true")
    parser.add_argument('-s', "--source", help='Stream source identifier', type=str, default=DEFAULT_SOURCE)

    args = parser.parse_args()
    return args

def main():
    """Prepares data for the object recognition demo"""
    args = getArgs()
    timesRestarted = 0
    pid = os.getpid()
    logger = Log(args.tdetect, args.tstat, args.tinfo, args.source, args.sendlogs, args.devo_server, args.devo_port)
    while True:
        process = Process(target=run, args = (args, pid, logger))
        signal.signal(signal.SIGUSR1, partial(signal_handler, process, logger))
        process.start()
        process.join()
        timesRestarted += 1
        logger.info("INFO", "RESTARTED ****************************************** %s" % (timesRestarted))


if __name__ == '__main__':
    main()
