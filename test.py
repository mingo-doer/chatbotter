import sys
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask,request
from test_chatterbot import SimpleChat
from flask import render_template

import os  #在顶头位置加上
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # ‘1’表示默认的显示等级，运行时显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # ‘2’运行时只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # ‘3’运行时只显示 Error

# def test(params, infos):
def test(params,infos):
    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    x_data, _ = pickle.load(open('D:/tfdemo/chatbot1.pkl', 'rb'))
    ws = pickle.load(open('D:/tfdemo/ws1.pkl', 'rb'))



    config = tf.ConfigProto(
        device_count = {'CPU':1, 'GPU':0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            x_test = [list(infos.lower())]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            for p in pred:
                ans = ws.inverse_transform(p)
                print(ans)
                return ans
cbot=SimpleChat()

app = Flask(__name__)
@app.route('/api/chatbot', methods=['get'])
def chatbot():
    infos = request.args['infos']
    res=cbot.get_response(infos)
    if res!='NO':
        return "".join(res)
    else:
        import json
        text = test(json.load(open('D:/tfdemo/params.json')), infos)
        # return text
        return "".join(text)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=8000)

def main():
    import json
    test(json.load(open('D:/tfdemo/params.json')))
#
# if __name__ == '__main__':
#     main()

