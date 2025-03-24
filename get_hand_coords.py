import requests
import numpy as np
import json
from pythonosc import udp_client

ip = "192.168.0.2"    
port = 1234    # Your desired port
osc_client = udp_client.SimpleUDPClient(ip, port)


def get_coord(ip,port=1234,osc_client= udp_client.SimpleUDPClient(ip, port)):

    #get hand coordinates from video stream header server
    # Connect to video stream over HTTP internet - so might have to change IP address day to day
    # url = "http://10.106.33.26:5000/video_feed"
    url = f"http://{ip}:5000/video_feed"
    print
    response = requests.get(url, stream=True)
    print(response)
    
    # if 404 break
    if response.status_code == 404:
        print("404!")
        raise Exception
        
        

    buffer = b""
    for chunk in response.iter_content(chunk_size=4096):
        buffer += chunk
        while b"--frame\r\n" in buffer:
            # Extract frame data
            _, buffer = buffer.split(b"--frame\r\n", 1)
            header, buffer = buffer.split(b"\r\n\r\n", 1)
            
            # Extract XYZ data from header
            if b"X-XYZ:" in header:
                xyz_data = header.split(b"X-XYZ: ")[1].split(b"\r\n")[0]
                xyz_json = xyz_data.decode("utf-8")
                print("Landmark 9 XYZ:", xyz_json)
                

                try:
                    data = json.loads(xyz_json)
                    x = data["x"]
                    y = data["y"]
                    z = data["z"]
                    
                    x = np.round(x,2)
                    y = np.round(y,2)
                    z = np.round(z,2)
                        
                except TypeError:
                    print("Error decoding JSON:", xyz_json)
                    x,y,z =0.0,0.0,0.0
                
                if x == None:
                    x = 0.0
                if y == None:
                    y = 0.0
                if z == None:
                    z = 0.0
                    
                print(x,y,z)
                
                
                
                print(f"Sending OSC: {x}, {y}, {z}")
                osc_client.send_message("/xyz", [x, y, z])
            
                value = "HI"
                # osc_client.send_message("/xyz", value)
                            
                return x,y,z
if __name__ == "__main__":
    # OSC setup
    
    while True:
        x,y,z = get_coord(ip)


