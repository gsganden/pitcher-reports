from flask import Flask, render_template, request, redirect
import requests
import pandas as pd
from datetime import datetime
from bokeh.plotting import figure, output_notebook, output_file, save
import os
import psycopg2
import urlparse

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        return render_template('results.html', result = run_query("SELECT * FROM pitches LIMIT 10;"))

def run_query(query):
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])

    conn = psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )

    cur = conn.cursor()

    cur.execute(query)
    result = cur.fetchone()

    cur.close()
    conn.close()

    return result

if __name__ == '__main__':
    app.run(port=33508)