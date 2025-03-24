from pythonosc import udp_client
from time import sleep

def get_message():
    return "Message from another script!"

ip = "10.106.33.26"    
port = 1234    # Your desired port
osc_client = udp_client.SimpleUDPClient(ip, port)

def trigger_unreal():
    """
    This function sends an OSC message to Unreal Engine
    """
    message_address = "/trigger"
    value = get_message()  # Value to be sent (you can change this as needed 
    # fetch message dynamically

    # Send the OSC message
    osc_client.send_message(message_address, value)
    print(f"Sent OSC message to {ip}:{port} with value:{value}")


while True:
    trigger_unreal()
    sleep(1)