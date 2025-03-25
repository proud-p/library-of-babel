#same as hand coord script

from pythonosc import dispatcher, osc_server

class HandCoordReceiver:
    def __init__(self):
        self.latest_coords = {"x": 0.0, "y": 0.0, "z": 0.0}

    def handle_xyz(self, address, *args):
        if len(args) == 3:
            self.latest_coords["x"] = round(args[0], 2)
            self.latest_coords["y"] = round(args[1], 2)
            self.latest_coords["z"] = round(args[2], 2)
            print(f"📥 Received on {address}: {args}")

    def get_coord(self):
        return (
            self.latest_coords["x"],
            self.latest_coords["y"],
            self.latest_coords["z"]
        )
        
    def handle_voice_prompt(self, address = "/voice_prompt", *args):
        text = " ".join(map(str, args)).strip()
        print(f"🆕 Received new prompt {address}: {text}")
        self.prompt = text  # update the prompt and regenerate text


    def start_receiver(self, ip="0.0.0.0", port=5009):
        disp = dispatcher.Dispatcher()
        disp.map("/xyz", self.handle_xyz)
        disp.map("/voice_prompt", self.handle_voice_prompt)

        server = osc_server.BlockingOSCUDPServer((ip, port), disp)
        print(f"🟢 OSC server listening on {ip}:{port}")
        server.serve_forever()
        



if __name__ == "__main__":
    receiver = HandCoordReceiver()
    wsl_ip= "172.30.40.252"
    receiver.start(ip=wsl_ip, port=5009)  # This blocks and listens forever
    x,y,z = receiver.get_coord()
    print(F"x: {x}, y: {y}, z: {z}")