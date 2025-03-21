import argparse
import math
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

# Custom handler to print /xyz messages
def print_xyz_handler(address, *args):
    print(f"Received OSC on {address}:", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The IP to listen on")
    parser.add_argument("--port", type=int, default=1234, help="The port to listen on")
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map("/xyz", print_xyz_handler)  # Map our OSC path to the handler

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    print("✅ Listening for OSC on {}:{}".format(args.ip, args.port))
    server.serve_forever()
