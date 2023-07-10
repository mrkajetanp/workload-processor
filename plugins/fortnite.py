import time
from wa import ApkWorkload, Parameter


class Fortnite(ApkWorkload):
    name = 'fortnite'
    package_names = ['com.epicgames.fortnite']
    activity = 'com.epicgames.unreal.GameActivity'
    view = "SurfaceView[com.epicgames.fortnite/com.epicgames.unreal.GameActivity](BLAST)"
    install_timeout = 200
    uninstall = False
    clear_data_on_reset = False
    description = """
    Fortnite
    """

    parameters = [
        Parameter('timeout', kind=int, default=126,
                  description='The amount of time the game should run for'),
    ]

    def setup(self, context):
        super(Fortnite, self).setup(context)
        # wait for the app to initialise
        time.sleep(30)

    def run(self, context):
        self.target.sleep(self.timeout)

    def update_output(self, context):
        super(Fortnite, self).update_output(context)

