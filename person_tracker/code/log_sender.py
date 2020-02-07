"""
Send logs to Devo platform
"""

import logging as log
import cv2 as cv
import numpy as np
import datetime
import base64
from utils.misc import COLOR_PALETTE

KEY = "CERTS/key.key"
CERT = "CERTS/cert.crt"
CHAIN = "CERTS/chain.crt"

### Devo Sender
from devo.sender import Sender, SenderConfigSSL, SenderConfigTCP


class Log:
    def __init__(self, tdetection, tstat, tinfo, source, sendlogs, devo_server, devo_port):
        self.tdetection = tdetection
        self.tstat = tstat
        self.tinfo = tinfo
        self.source = source
        self.sendlogs = sendlogs
        if sendlogs:
            self.sender = self._set_sender(devo_server, devo_port)

    def _set_sender(self,  devo_server, devo_port):
        engine_config = SenderConfigSSL(address=(devo_server, devo_port), key=KEY, cert=CERT,
                                            chain=CHAIN)
        return Sender(engine_config)

    def detections(self, frames, detections):
        self._log_detection(self.tdetection, frames, detections)

    def stats(self, stats):
        self._log_stat(self.tstat, stats)

    def info(self, type, msg):
        self._log_info(self.tinfo, type, msg)

    def _sendMsg(self, table, msg):
        try:
            if self.sendlogs:
                self.sender.send(tag=table, msg=msg)
                #print(msg)
        except Exception as e:
            self.info("ERROR", "***LOSING EVENT*** " + str(e))

    def draw_detection(frame, obj):
        """Draws detections and labels"""

        left, top, right, bottom = obj.rect
        id = obj.label
        label = 'ID ' + str(id)
        box_color = COLOR_PALETTE[id % len(COLOR_PALETTE)]

        cv.rectangle(frame, (left, top), (right, bottom),
                    box_color, thickness=3)
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def _build_msg(self, msg, obj, frame):
        left, top, right, bottom = obj.rect
        msg += '|' + str(left) + '|' + str(top) + '|' + str(right) + '|' + str(bottom)
        msg += '|' + str(obj.label)


        #use in case that you like to send the image too
        #if labels_map[obj[1]].upper() in objectsToLog:
        #   msg += '|' + self._b64box_msg(top, bottom, left, right, frame)  # just the person
        #else:
        #    msg += '|' + "null"

        return msg

    def _b64box_msg(self, top, bottom, left, right, frame):
        frameBox = np.copy(frame)
        crop_img = frameBox[top:bottom, left:right]
        retval, buffer = cv.imencode('.jpg', crop_img)
        jpg_as_text = base64.b64encode(buffer)
        encodedStr = str(jpg_as_text, "utf-8")
        return 'jpg;base64; ' + encodedStr

    def __log_detections(self, table, cam, frame, detections, dateTimeObj):
        """Draws detections and labels"""
        msg = "cam_" + str(cam) + "|" + str(frame.shape[0]) + "|" + str(frame.shape[1]) + "|" + self.source \
            + "|" + dateTimeObj.strftime("%Y/%m/%d %H:%M:%S")

        for i, obj in enumerate(detections):
            msg1 = self._build_msg(msg, obj, frame)
            self._sendMsg(table, msg1)

    def _log_detection(self, table,  frames, all_objects):
        assert len(frames) == len(all_objects)
        cam=0
        dateTimeObj = datetime.datetime.now()
        for frame, objects in zip(frames, all_objects):
            self.__log_detections(table, cam, frame, objects, dateTimeObj)
            cam=cam+1

    def _log_stat(self, table, stats):
        msg_stat = self.source + "|" + str((stats['cap.end'] - stats['cap.start']) * 1000) \
                   + "|" + str((stats['inference.end'] - stats['inference.start']) * 1000) \
                   + "|" + str((stats['end'] - stats['start']) * 1000)
        self._sendMsg(table, msg_stat)

    def _log_info(self, table, type, msg):
        msg_info = self.source + "|" + type + "|" + msg
        log.info(msg_info)
        self._sendMsg(table, msg_info)
