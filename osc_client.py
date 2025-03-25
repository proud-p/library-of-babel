from pythonosc import udp_client
from time import sleep


class OSCSender:
    def __init__(self, ip="127.0.0.1",port=1234):
        global osc_client
        osc_client = udp_client.SimpleUDPClient(ip, port)
        print(f"OSC client to send messages initialized at {ip}:{port}")
        
    def send_osc_message(self, message, message_address = "/answer"):
        
        def get_message(message):
            #TODO clean messages
            return message
        
        global osc_client
        if osc_client is None:
            raise RuntimeError("OSC client not initialized. Call init_osc(ip, port) first.")
        

        value = get_message(message)  # This should return a string or number
        osc_client.send_message(message_address, value)
        print(f"Sent OSC message to {osc_client._address}:{osc_client._port} → {value}")
        
    
            
            
            
        

if __name__ == "__main__":
    while True:
        ip = "192.168.0.2"        
        port = 1234    # Your desired port
        Sender = OSCSender(ip, port=1234)
        Sender.send_osc_message("message from wsl!")
        sleep(60)