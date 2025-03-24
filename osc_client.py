from pythonosc import udp_client
from time import sleep

def get_message():
    return "Message from another script!"


osc_client = None  # global reference to reuse the client

def init_osc(ip="127.0.0.1", port=1234):
    global osc_client
    osc_client = udp_client.SimpleUDPClient(ip, port)
    print(f"OSC client initialized at {ip}:{port}")

def send_osc_message():
    """
    Send OSC message to Unreal Engine using existing client.
    """
    global osc_client
    if osc_client is None:
        raise RuntimeError("OSC client not initialized. Call init_osc(ip, port) first.")
    
    message_address = "/answer"
    value = get_message()  # This should return a string or number
    osc_client.send_message(message_address, value)
    print(f"Sent OSC message to {osc_client._address}:{osc_client._port} → {value}")


if __name__ == "__main__":
    while True:
        ip = "192.168.0.2"        
        port = 1234    # Your desired port
        init_osc(ip)
        osc_client = udp_client.SimpleUDPClient(ip, port)
        send_osc_message()
        sleep(1)