from typing import Literal


class MainModule:
    def __init__(self, env: Literal['dev', 'prod'] = "prod", computer: bool = True, lineFollow: bool = True,
                 largeObject: bool = True,
                 splitRoad: bool = True, yolo: bool = True, serial: bool = True):
        self.env = env
        self.computer = computer
        self.lineFollow = lineFollow
        self.largeObject = largeObject
        self.splitRoad = splitRoad
        self.yolo = yolo
        self.serial = serial

    def mPrint(self, data):
        if self.env == "dev":
            print(data)
