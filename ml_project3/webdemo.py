import os
import sys
import torch
from flask import Flask, request, render_template
import argparse
from module import Tokenizer, init_model_by_key

MODEL_PATH = sys.argv[1]
class Context(object):
    def __init__(self, path):
        print(f"loading pretrained model from {path}")
        self.device = torch.device('cpu')
        model_info = torch.load(path)
        self.tokenizer = model_info.get('tokenizer', model_info.get('tokenzier'))
        self.model = init_model_by_key(model_info['args'], self.tokenizer)
        self.model.load_state_dict(model_info['model'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, s):
        input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
        source_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            pred = self.model.generate(input_ids, source_mask, max(1, input_ids.size(1) * 2)).squeeze(0).tolist()
        pred = self.tokenizer.decode(pred)
        return pred
        
app = Flask(__name__)
ctx = Context(MODEL_PATH)

@app.route('/<coupletup>')
def api(coupletup):
    return ctx.predict(coupletup)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    coupletup = request.form.get("coupletup")
    coupletdown = ctx.predict(coupletup)
    return render_template("index.html", coupletdown=coupletdown)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
