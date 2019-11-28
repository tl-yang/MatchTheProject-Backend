import os
import io
import base64
import time

import numpy as np
import PIL.ImageOps
from PIL import Image
from flask import Flask
from flask import request
from flask_cors import CORS
from recognizer.svc_recognizer import Recognizer, IMAGE_WIDTH, IMAGE_HEIGHT
# from recognizer.cnn_model import CnnRecognizer, FLAGS
from recognizer.cnn9_model import Cnn9Model, IMAGE_SIZE, LABEL_MAP

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'log'))


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)

    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    recognizer = Recognizer()
    # cnn_recognizer = CnnRecognizer()
    model = Cnn9Model()

    # recognition function
    @app.route('/recognition', methods=('POST',))
    def recognize():
        body = request.form
        st = time.time()
        # x = base64.b64decode(body['input'])
        level = body['level']
        input_image = Image.frombytes('RGBA', (int(body['width']), int(body['height'])),
                                      request.files['input'].stream.read())
        # input_image = Image.frombytes('RGBA', (body['width'], body['height']), x)
        # if body['method'] == 'cnn':
        #     preprocessed_image = input_image.resize((FLAGS.image_size, FLAGS.image_size)).convert('L')
        #     preprocessed_image = PIL.ImageOps.invert(preprocessed_image)
        #     prediction, pred_prob = cnn_recognizer.produce(preprocessed_image)
        #     print(prediction)
        #     result = [int(prediction.flatten()[0])]
        if body['method'] == '9cnn':
            preprocessed_image = input_image.convert('L').resize(IMAGE_SIZE)
            preprocessed_image = PIL.ImageOps.invert(preprocessed_image)
            top_k_pred = model.produce(image=preprocessed_image).flatten()
            result = [LABEL_MAP[pred] if pred in LABEL_MAP else -1 for pred in top_k_pred]
        else:
            preprocessed_image = input_image.resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT)).convert('L').point(lambda x: 255 if x > 10 else 0, mode='1')
            im = np.array(preprocessed_image).reshape(1, -1)
            result = recognizer.produce(im, level)

        log_image_name = '{}_{}.jpg'.format(level, time.time())
        original_image_name = 'original_{}_{}.png'.format(level, time.time())
        with open(os.path.join(log_dir, log_image_name), 'wb+') as f:
            preprocessed_image.save(f)
        with open(os.path.join(log_dir, original_image_name), 'wb+') as f:
            input_image.save(f)
        print(time.time() - st)
        return {'predictions': result}

    return app
