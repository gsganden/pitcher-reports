from flask import Flask, render_template, request, redirect, make_response
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from bokeh.plotting import figure, output_file, save
from bokeh.models import Range1d, FixedTicker
from bokeh.io import gridplot
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sqlalchemy
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.mixture import GMM
from matplotlib.ticker import NullFormatter


app = Flask(__name__)
import os
import urlparse

urlparse.uses_netloc.append("postgres")
url = urlparse.urlparse(os.environ["DATABASE_URL"])

database = url.path,
user = url.username,
password = url.password,
host = url.hostname,
port = url.port
scheme = url.scheme

engine = sqlalchemy.create_engine('%s://%s:%s@%s:%s/%s' %
                                  (scheme, user[0], password[0], host[0],
                                   port, database[0][1:]))

@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET': 
        return render_template('index.html')
    else:
        pitcher = request.form['pitcher']
        season = request.form['season']
        eliasid, throws = get_eliasid_throws(pitcher)
        data = get_data(pitcher, season, eliasid, throws)
        return render_template('results.html',
                               movement_plot=plot_movement(data, pitcher, season),
                               pitcher=pitcher,
                               season=season)


def get_eliasid_throws(pitcher):
    query = '''
            SELECT eliasid, throws
            FROM players
            WHERE first = '%s'
            AND last = '%s'
            ''' % (pitcher.split()[0], pitcher.split()[1])

    return pd.read_sql(query, engine).values[0]


def get_data(pitcher_name, season, eliasid, throws):
    query = '''
        SELECT pitch_type,
            start_speed,
            pfx_x,
            pfx_z,
            stand
        FROM pitches
        JOIN (
                SELECT *
                FROM atbats
                WHERE pitcher = %s
            ) atbats
        ON pitches.ab_id = atbats.ab_id
        WHERE start_speed IS NOT NULL
            AND pitches.des != 'Intent Ball'
            AND sv_id > '%s0000_000000'
            AND sv_id < '%s0000_000000'
        ''' % (eliasid,
               season[-2:],
               str(int(season) + 1)[-2:])

    return pd.read_sql(query, engine)


def plot_movement(data, pitcher, season):
    gaussians = []

    pitch_types = sorted(list(data['pitch_type'].unique()))
    pitch_type_counts = data.groupby('pitch_type').size()

    df = data
    for pitch_type in pitch_types:
        if float(df[df['pitch_type'] == pitch_type].shape[0]) / df.shape[0] < .02:
            continue
        
        gmm = GMM(covariance_type = 'full')
        
        sub_df = df[df['pitch_type'] == pitch_type][['pfx_x', 'pfx_z', 'start_speed']]
        gmm.fit(sub_df)

        
        x = np.arange(-20, 20, 0.25)
        y = np.arange(-20, 20, 0.25)
        X, Y = np.meshgrid(x, y)
        
        gaussians.append(plt.mlab.bivariate_normal(X, Y, sigmax=np.sqrt(gmm._get_covars()[0][0][0]), 
                                         sigmay=np.sqrt(gmm._get_covars()[0][1][1]), 
                                         sigmaxy=gmm._get_covars()[0][0][1], 
                                         mux=gmm.means_[0][0], 
                                         muy=gmm.means_[0][1]))

    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25]) 

    ax1 = plt.subplot(gs[0])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'], \
                    alpha=.3, cmap='inferno', norm = Normalize(70, 100))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.ylabel('Vertical break')
    plt.title('All')
    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha = .3)

    df = data[data['stand'] == 'R']
    ax2 = plt.subplot(gs[1])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'], \
                    alpha=.3, cmap='inferno', norm = Normalize(70, 100))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.xlabel('Horizontal break, catcher\'s perspective')
    plt.title('Batter right-handed')
    ax2.yaxis.set_major_formatter( NullFormatter() )
    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha = .3)


    df = data[data['stand'] == 'L']
    ax3 = plt.subplot(gs[2])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'], \
                    alpha=.3, cmap='inferno', norm = Normalize(70, 100))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.colorbar().set_label('Velocity')
    plt.title('Batter left-handed')
    ax3.yaxis.set_major_formatter( NullFormatter() )
    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha = .3)

    plt.tight_layout()

    # Make Matplotlib write to BytesIO file object and grab
    # return the object's string
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def get_results(results_file):
    soup = BeautifulSoup(results_file, 'html.parser')
    contents = ''
    for item in soup.body.contents:
        contents += '\n' + unicode(item)
    return contents

if __name__ == '__main__':
    app.run(port=33507, debug=False)
