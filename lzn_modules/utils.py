from threading import Timer
import atexit
import time


class TimeMaster(object):
    count = 0
    timer = None

    @staticmethod
    def repeat(rep, delay, func, setStatus, *args):
        "repeat func rep times with a delay given in seconds"
        if TimeMaster.count < rep:
            # call func, you might want to add args here
            func(*args)
            TimeMaster.count += 1
            # setup a timer which calls repeat recursively
            # again, if you need args for func, you have to add them here
            TimeMaster.timer = Timer(delay, TimeMaster.repeat, (rep, delay, func, setStatus, *args))
            # register timer.cancel to stop the timer when you exit the interpreter
            atexit.register(TimeMaster.timer.cancel)
            TimeMaster.timer.start()
        else:
            setStatus('rejected')

    @staticmethod
    def stop():
        # TimeMaster.count = 99999
        TimeMaster.timer.cancel()

    @staticmethod
    def setTimeout(func, delay, *argments):
        Timer(delay, func, argments).start()
