import requests
import cv2
import numpy as np

#get hand coordinates from video stream header server
# Connect to video stream
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

        # Decode image
        img_array = np.frombuffer(buffer, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Show video
        print(xyz_json)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
