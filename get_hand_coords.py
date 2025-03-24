from pythonosc import dispatcher, osc_server

# Store latest coords
latest_coords = {"x": 0.0, "y": 0.0, "z": 0.0}

def handle_xyz(addr, *args):
    if len(args) == 3:
        latest_coords["x"] = round(args[0], 2)
        latest_coords["y"] = round(args[1], 2)
        latest_coords["z"] = round(args[2], 2)
        print(f"📥 Received /xyz → {latest_coords['x']}, {latest_coords['y']}, {latest_coords['z']}")

def get_coord():
    return latest_coords["x"], latest_coords["y"], latest_coords["z"]

def start_osc_coord_receiver(ip="0.0.0.0", port=1234):
    disp = dispatcher.Dispatcher()
    disp.map("/xyz", handle_xyz)
    server = osc_server.BlockingOSCUDPServer((ip, port), disp)
    print(f"🟢 Listening for OSC /xyz on {ip}:{port} ...")
    server.serve_forever()


if __name__ == "__main__":
    ip ="127.0.0.1"
    osc_port = 5009
    start_osc_coord_receiver(ip,osc_port)
    