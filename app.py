from flask import Flask, render_template, request, redirect
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
import sqlalchemy
import os
import psycopg2
import urlparse
from bokeh.plotting import figure, output_notebook, output_file, save
from bokeh.models import Range1d, FixedTicker
from bokeh.io import gridplot
from bs4 import BeautifulSoup
import sqlalchemy
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
    pitcher = request.form['pitcher']
    return render_template('results.html', plot = get_results(plot(get_data(pitcher))), pitcher = pitcher)

def get_data(pitcher_name):
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ["DATABASE_URL"])

    database=url.path,
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
    scheme = url.scheme

    engine = sqlalchemy.create_engine('%s://%s:%s@%s:%s/%s' % (scheme, user[0], password[0], host[0], port, database[0][1:]))
    conn = engine.connect()

    query = '''
        SELECT pitch_type,
            start_speed,
            pfx_x,
            pfx_z
        FROM pitches
        JOIN (
                SELECT *
                FROM atbats
                WHERE pitcher = (
                                SELECT eliasid
                                FROM players
                                WHERE first = '%s'
                                AND last = '%s'
                                )
            ) atbats
        ON pitches.ab_id = atbats.ab_id
        WHERE start_speed IS NOT NULL
            AND pitches.des != 'Intent Ball'
            AND sv_id > '150000_000000'
        ''' % (pitcher_name.split()[0], pitcher_name.split()[1])
    return pd.read_sql(query, engine)

def plot(df):
    
    norm = Normalize(70, 100)

    colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, 
              _ in 255*plt.cm.inferno(norm(df['start_speed']))]

    TOOLS = 'resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select'

    p = figure(tools = TOOLS)

    p.x_range = Range1d(start=-20, end=20)
    p.y_range = Range1d(start=-20, end=20)
    p.xaxis.axis_label = 'Horizontal movement (catcher\'s perspective)'
    p.yaxis.axis_label = 'Vertical movement'

    p.scatter(df['pfx_x'], df['pfx_z'],
              fill_color=colors, fill_alpha=0.6,
              line_color=None)

    y = np.linspace(70.8, 99.2, 1000)
    x = np.zeros(1000)

    colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, 
              _ in 255*plt.cm.inferno(Normalize()(y))]

    q = figure(width = 140)
    q.x_range = Range1d(start=-.5, end=.5)
    q.y_range = Range1d(start=70, end=100)
    q.xaxis.ticker = FixedTicker(ticks = [])
    q.yaxis.axis_label = 'Velocity (mph)'

    q.scatter(x, y, fill_color=colors, alpha = 0.3, marker = 'square', line_color=None, size = 30)

    r = gridplot([[p, q]])

    output_file('results.html')

    save(r)

    return open('results.html')

def get_results(results_file):

    soup = BeautifulSoup(results_file, 'html.parser')
    contents = ''
    for item in soup.body.contents:
        contents += '\n' + unicode(item)
    return contents

if __name__ == '__main__':
  app.run(port=33507)