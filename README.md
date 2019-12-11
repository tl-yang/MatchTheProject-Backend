# MatchTheProject-Backend
Visit Frontend [here](https://github.com/tl-yang/MatchTheChar)

## Dependencies:

- numpy
- Pillow
- Flask
- Flask-Cors
- tensorflow == 1.14

## Pre-trained Models
Download [Here](https://drive.google.com/open?id=1dmHhyawT0GplbWmmeFgtUfFwm0UVpiF_)
and put it in `recognizer/tf_cnn9_checkpoints`. Other models are also provided for your reference.
## How to Run:
``````
export FLASK_ENV=development 
export FLASK_APP=recognizer
export FLASK_DEBUG=1
flask run
``````
The server will be serve on `http://127.0.0.1:5000`.  
The recognition api in on `/recognition` using `POST` with Form Data:
```
width: 191
height: 152
input: (binary)
method: 9cnn
level: level1
``` 
And the response is a json string of predictions: 
```
{"predictions":[-1,-1,-1,6,-1,-1,9,-1,-1,-1]}
```
