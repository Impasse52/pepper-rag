#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: A Simple class to get & read FaceDetected Events"""

import sys
import time

import qi

# connection settings
pepper_ip = "192.168.1.3"
pepper_port = 9559
tablet_ip = "198.18.0.1"

targetName = "Face"
faceWidth = 0.1


class Pepper(object):
    def __init__(self, app):
        super(Pepper, self).__init__()

        app.start()
        session = app.session

        # Get the service ALMemory.
        self.memory = session.service("ALMemory")

        self.animation_player_service = session.service("ALAnimationPlayer")

        self.face_service = session.service("ALFaceDetection")
        self.face_service.enableTracking(True)

        self.tabletService = session.service("ALTabletService")
        self.tabletService.hideImage()

        self.serviceAnimatedSpeech = session.service("ALAnimatedSpeech")

        self.serviceTextToSpeech = session.service("ALTextToSpeech")
        self.serviceTextToSpeech.setParameter("speed", 90)

        self.serviceLed = session.service("ALLeds")
        self.serviceMotion = session.service("ALMotion")
        self.serviceMemory = session.service("ALMemory")
        self.servicePosture = session.service("ALRobotPosture")
        self.serviceTouch = session.service("ALTouch")
        self.serviceAudioPlayer = session.service("ALAudioPlayer")

        self.servicePosture.goToPosture("StandInit", 0.5)

    def run(self):
        """
        Loop on, wait for events until manual interruption.
        """
        print("Starting PepperProva")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping PepperProva")
            sys.exit(0)

    def tts(self, text: str):
        print(f"tts: {text}")
        self.serviceTextToSpeech.say(text)


def init_pepper(ip: str, port: int):
    try:
        connection_url = "tcp://" + ip + ":" + str(port)
        app = qi.Application(["PepperProva", "--qi-url=" + connection_url])
        app.start()

        pepper = Pepper(app)
        # pepper.run()

        return pepper
    except RuntimeError:
        print(
            "Can't connect to Naoqi at ip \"" + ip + '" on port ' + str(port) + ".\n"
            "Please check your script arguments. Run with -h option for help."
        )
        sys.exit(1)


if __name__ == "__main__":
    pepper = init_pepper(pepper_ip, pepper_port)

    pepper.tts("prova")
