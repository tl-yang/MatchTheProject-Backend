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

    # recognition function
    @app.route('/recognition', methods=('POST',))
    def recognize():
        st = time.time()
        body = request.get_json()
        x = base64.b64decode(body['input'])
        level = body['level']
        # im = np.array(Image.open(io.BytesIO(x))).reshape(1, -1)
        original_image = Image.frombytes('RGBA', (body['width'], body['height']), x)
        image_bytes = original_image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT)).convert('L')
        inverted_image = PIL.ImageOps.invert(image_bytes)
            # .point(lambda x: 255 if x > 10 else 0, mode='1')
        log_image_name = '{}_{}.jpg'.format(level, time.time())
        inverted_image_name = 'inverted_{}_{}.jpg'.format(level, time.time())
        original_image_name = 'original_{}_{}.png'.format(level, time.time())
        with open(os.path.join(log_dir, log_image_name), 'wb+') as f:
            image_bytes.save(f)
        with open(os.path.join(log_dir, original_image_name), 'wb+') as f:
            original_image.save(f)
        with open(os.path.join(log_dir, inverted_image_name), 'wb+') as f:
            inverted_image.save(f)
        im = np.array(image_bytes).reshape(1, -1)
        result = recognizer.produce(im, level)
        print(time.time() - st)
        return {'prediction': result}

    return app
