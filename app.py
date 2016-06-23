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
import os
import urlparse


app = Flask(__name__)

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

pitch_type_dict = dict(FA = 'Fastball',
                       FF = '4-Seam Fastball',
                       FT = '2-Seam Fastball',
                       FC = 'Cutter',
                       SL = 'Slider',
                       CH = 'Changeup',
                       CU = 'Curveball',
                       KC = 'Knuckle-Curve',
                       KN = 'Knuckleball',
                       EP = 'Eephus')


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
                               movement_plot=plot_movement(data),
                               selection_plot=plot_selection(data),
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
        SELECT 
            CASE
                WHEN pitch_type = 'SF' OR pitch_type = 'FS' OR pitch_type = 'SI'
                THEN 'FT'
                WHEN pitch_type = 'CB'
                THEN 'CU'
                ELSE pitch_type
            END AS pitch_type,
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


def plot_movement(data):
    gaussians = []
    gmms = []

    pitch_types = sorted(list(data['pitch_type'].unique()))
    pitch_type_counts = data.groupby('pitch_type').size()

    df = data
    filtered_pitch_types = pitch_types[:]
    pitch_type_counts = data.groupby('pitch_type').size()

    for pitch_type in pitch_types:
        if float(df[df['pitch_type'] == pitch_type].shape[0]) / df.shape[0] < .02:
            filtered_pitch_types.remove(pitch_type)
            
    for pitch_type in filtered_pitch_types:
        gmm = GMM(covariance_type = 'full')
        
        sub_df = df[df['pitch_type'] == pitch_type][['pfx_x', 'pfx_z', 'start_speed']]
        gmm.fit(sub_df)

        
        x = np.arange(-20, 20, 0.25)
        y = np.arange(-20, 20, 0.25)
        X, Y = np.meshgrid(x, y)

        gmms.append(gmm)
        
        gaussians.append(plt.mlab.bivariate_normal(X, Y, sigmax=np.sqrt(gmm._get_covars()[0][0][0]), 
                                         sigmay=np.sqrt(gmm._get_covars()[0][1][1]), 
                                         sigmaxy=gmm._get_covars()[0][0][1], 
                                         mux=gmm.means_[0][0], 
                                         muy=gmm.means_[0][1]))

    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25]) 

    ax1 = plt.subplot(gs[0])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'],
                alpha=.3, cmap='inferno', norm = Normalize(70, 100))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.ylabel('Vertical break')
    plt.title('All batters')
    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha = .3)  
        ax1.text(gmms[index].means_[0][0], 
                 gmms[index].means_[0][1], 
                 pitch_type_dict[filtered_pitch_types[index]],
                 ha='center', 
                 va='center',
                 color='k',
                 size=10)

    df = data[data['stand'] == 'R']
    ax2 = plt.subplot(gs[1])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'],
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
        ax2.text(gmms[index].means_[0][0], 
                 gmms[index].means_[0][1], 
                 pitch_type_dict[filtered_pitch_types[index]],
                 ha='center', 
                 va='center',
                 color='k',
                 size=10)

    df = data[data['stand'] == 'L']
    ax3 = plt.subplot(gs[2])
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'],
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
        ax3.text(gmms[index].means_[0][0], 
                 gmms[index].means_[0][1], 
                 pitch_type_dict[filtered_pitch_types[index]],
                 ha='center', 
                 va='center',
                 color='k',
                 size=10)

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

def plot_selection(data):

    fig = plt.figure(figsize=(12,4))
    ax = fig.gca()

    pitch_type_counts = data.groupby('pitch_type').size()
    filtered_pitch_types = pitch_type_counts[pitch_type_counts > .02 * sum(pitch_type_counts)]
    num_pitches_to_righties = float(data[data['stand'] == 'R'].shape[0])
    num_pitches_to_lefties = float(data[data['stand'] == 'L'].shape[0])

    for index, pitch_type in enumerate(list(data[data['pitch_type'].isin(filtered_pitch_types.index.values)]\
                                             .groupby('pitch_type')\
                                             .agg({'start_speed': [np.size, np.mean]})['start_speed']\
                                             .sort_values(by='mean')
                                             .index)):
        pitch_data = data[data['pitch_type'] == pitch_type].groupby(['pitch_type', 'stand']).size()
        plt.scatter(index, pitch_data[pitch_type]['R']/num_pitches_to_righties, color='r', marker='$R$', s=40)
        plt.scatter(index, pitch_data[pitch_type]['L']/num_pitches_to_lefties, color='b', marker='$L$', s=40)
        ax.text(index, -.05, pitch_type_dict[pitch_type], ha='center')

    plt.ylim([0, 1])
    plt.xticks([])
    plt.title('Pitch distribution by batter handedness, in order of increasing velocity')

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
    app.run(port=33507, debug=True)
