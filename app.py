import cv2
import jsonpickle
import numpy as np
from deepface import DeepFace
from flask import Flask, request, Response

app = Flask(__name__)


@app.route('/api/test', methods=['POST'])
def test():
    print("request recieved")
    r = request
    sstring = r.data
    imageString = r.data.__str__().split("=")[1]
    #imageString = r.data.decode("UTF-8").split("=")[1]
    # convert string of image data to uint8
    nparr = np.fromstring(imageString, np.uint8)
    print("data transformed")
    try:
        img = cv2.imdecode(nparr, 0)
    except:
        print("data transformed")
    if img is not None:
        emotionalities = DeepFace.analyze(img, tuple('emotion'))
        print(emotionalities)
        response = {'message': emotionalities[0]}
        response_pickled = jsonpickle.encode(response)
    else:
        response_pickled = jsonpickle.encode({'message': 'image was empty'})
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
