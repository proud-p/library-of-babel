import requests
import numpy as np
import json


def get_coord():

    #get hand coordinates from video stream header server
    # Connect to video stream over HTTP internet - so might have to change IP address day to day
    url = "http://10.106.33.26:5000/video_feed"
    response = requests.get(url, stream=True)

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
                except TypeError:
                    print("Error decoding JSON:", xyz_json)
                    x,y,z =0.0,0.0,0.0
                
                if x == None:
                    x = 0.0
                if y == None:
                    y = 0.0
                if z == None:
                    z = 0.0
                            
                return x,y,z
            
if __name__ == "__main__":
    while True:
        x,y,z = get_coord()


