from flask import Flask, render_template, request, redirect
import requests
import pandas as pd
from datetime import datetime
from bokeh.plotting import figure, output_notebook, output_file, save

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        pitcher = request.form['pitcher']
        image_file = pitcher.lower()
        image_file = image_file.split()
        image_file = '_'.join(image_file) + '.png'
        return render_template('results.html', pitcher = pitcher, image_file = image_file)

if __name__ == '__main__':
    app.run(port=33508)