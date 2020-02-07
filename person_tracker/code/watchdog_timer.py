from threading import Thread, Event
from datetime import datetime, timezone
from time import sleep
import logging


logger = logging.getLogger()


class WDT:
    def __init__(self, logger, callback, check_interval_sec: float = 0.01, trigger_delta_sec: float = 1, identifier=None):
        self.check_interval_sec = check_interval_sec
        self.trigger_delta = trigger_delta_sec * 1000000
        self.callback = callback
        self.identifier = identifier
        self.logger = logger

        self.pause_flag = False
        self.resume_event = Event()
        self.thread_stopped = Event()
        self.poison_pill = False

        self.internal_ts = self.now()
        self.checker_thread_name = 'wdt_checker'
        self.start()
        self.logger.info("INFO",'WDT: Started')

    def __del__(self):
        self.stop()
        self.logger.info("INFO", 'WDT: Destroyed through destructor')

    @staticmethod
    def now():
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000000)

    def checker(self):
        while True:
            self.logger.info("DEBUG", 'WDT: Running checker')

            if self.poison_pill:
                break

            if self.pause_flag:
                self.logger.info("DEBUG", 'WDT: The {} thread is now paused')
                self.resume_event.wait()
                self.resume_event.clear()
                self.logger.info("DEBUG",'WDT: The {} thread has now resumed')

            elif (self.now() - self.internal_ts) > self.trigger_delta:
                if self.identifier is None:
                    self.pause()
                    logger.info("*** BLOQUEADO, calling to recover ****")
                    self.callback()
                    self.resume()
                else:
                    self.callback(self, self.identifier)

            #logger.debug('WDT: Sleeping for {} sec.'.format(self.check_interval_sec))
            sleep(self.check_interval_sec)

        #logger.debug('WDT: The checker thread has exited the loop')
        self.thread_stopped.set()

    def update(self):
        self.internal_ts = self.now()
        #logger.debug('WDT: Internal timer was set to: {}'.format(self.internal_ts))

    def reset(self):
        self.update()

    def pause(self):
        self.pause_flag = True
        #logger.debug('WDT: Pause command was invoked')

    def resume(self):
        self.pause_flag = False
        #logger.debug('WDT: Resume command was invoked')
        self.resume_event.set()

    def start(self):
        self.update()

        try:
            t = Thread(name=self.checker_thread_name, target=self.checker)
            t.start()
        except Exception as exp:
            self.logger.info("ERROR",'WDT: Unable to launch the {} thread'.format(self.checker_thread_name))
            self.logger.info("ERROR",'Exception: {}'.format(exp))

    def stop(self):
        self.poison_pill = True
        #logger.debug('WDT: Poison pill was injected, to stop the {} thread'.format(self.checker_thread_name))
        self.thread_stopped.wait()
        self.logger.info("ERROR",'WDT: Stopped')

    def get_internal_time(self):
        return self.internal_ts