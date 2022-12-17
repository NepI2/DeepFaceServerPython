import base64

import cv2
import jsonpickle
import urllib.parse
import numpy as np
from PIL import Image
import mat
from deepface import DeepFace
from matplotlib import pyplot as plt
from urllib.parse import urlparse
from flask import Flask, request, Response


app = Flask(__name__)


@app.route('/api/test', methods=['POST'])
def test():
    print("request recieved")
    r = request
    data = urllib.parse.unquote_plus(r.data.decode(encoding='UTF-8', errors='strict'))
    image = data[5:len(data)]
    # imageString = r.data.decode("UTF-8").split("=")[1]
    # convert string of image data to uint8
    jpg_original = base64.b64decode(image)
    nparr = np.frombuffer(jpg_original, dtype=np.uint8)
    #nparr = np.frombuffer(image, dtype="uint8")
    # nparr = mat.Mat.from_string(image)

    # img = Image.fromarray(nparr, 'RGB')
    # img.show()
    plt.imshow(nparr, interpolation='nearest')
    plt.show()
    print("data transformed")
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        print("data transformed")
    if img is not None:
        emotionalities = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        print(emotionalities)
        response = {'message': emotionalities}
        response_pickled = jsonpickle.encode(response)
    else:
        response_pickled = jsonpickle.encode({'message': 'image was empty'})
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
