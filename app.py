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

pitch_type_dict = dict(FA = 'fastball',
                       FF = 'four-seam fastball',
                       FT = 'two-seam fastball',
                       FC = 'fastball (cutter)',
                       FS = 'fastball (sinker, split-fingered)',
                       SI = 'fastball (sinker, split-fingered)',
                       SF = 'fastball (sinker, split-fingered)',
                       SL = 'slider',
                       CH = 'changeup',
                       CB = 'curveball',
                       CU = 'curveball',
                       KC = 'knuckle-curve',
                       KN = 'knuckleball',
                       EP = 'eephus')

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
        movement_plot=plot_movement(data)
        return render_template('results.html',
                               movement_plot=movement_plot,
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
            stand,
            pitches.ball,
            pitches.strike
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

    fig = plt.figure(figsize=(8,6))
    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'],
            alpha=.3, cmap='inferno', norm = Normalize(70, 100), s=10)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.ylabel('Vertical Break (Inches)')
    plt.xlabel('Horizontal Break (Inches)   ')
    plt.colorbar().set_label('Velocity')
    ax = plt.gca()
    ax.text(.65, .98, ''.join([pitch_type + ': ' + pitch_type_dict[pitch_type] + '\n' 
                            for pitch_type in filtered_pitch_types])[:-1],
                            horizontalalignment='left',
                            verticalalignment='top',
                         transform=ax.transAxes)\
                   .set_bbox(dict(color='w', alpha=0.3, edgecolor='k'))

    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha = .3)  
        ax.text(gmms[index].means_[0][0], 
                 gmms[index].means_[0][1], 
                 filtered_pitch_types[index],
                 ha='center', 
                 va='center',
                 color='k',
                 size=10,
                 backgroundcolor='w')\
                    .set_bbox(dict(color='w', alpha=0.3, edgecolor='k'))

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
    plt.figure(figsize=(8,8))

    pitch_type_counts = data.groupby('pitch_type').size()
    filtered_pitch_types = pitch_type_counts[pitch_type_counts > .02 * sum(pitch_type_counts)]
    pitch_type_list = list(data[data['pitch_type'].isin(filtered_pitch_types.index.values)]\
                                             .groupby('pitch_type')\
                                             .agg({'start_speed': [np.size, np.mean]})['start_speed']\
                                             .sort_values(by='mean')
                                             .index)

    for plot_num in range(1,21):
        plt.subplot(5,4,plot_num)
        num_balls = ((plot_num - 1) // 4) - 1
        num_strikes = ((plot_num - 1) % 4) - 1
        pitch_data = data.copy()
        if num_balls > -1:
            pitch_data = pitch_data[pitch_data['ball'] == str(num_balls)]
        if num_strikes > -1:
            pitch_data = pitch_data[pitch_data['strike'] == str(num_strikes)]
        num_pitches_to_righties = float(pitch_data[pitch_data['stand'] == 'R'].shape[0])
        num_pitches_to_lefties = float(pitch_data[pitch_data['stand'] == 'L'].shape[0])
        for index, pitch_type in enumerate(pitch_type_list):
            filter_pitch_data = pitch_data[pitch_data['pitch_type'] == pitch_type]
            try:
                plt.scatter(index, 
                            filter_pitch_data[filter_pitch_data['stand'] == 'R'].shape[0]/num_pitches_to_righties, 
                            color='r', alpha=.5, s=10, marker = '$R$')
            except ZeroDivisionError:
                pass
            try:
                plt.scatter(index, 
                            filter_pitch_data[filter_pitch_data['stand'] == 'L'].shape[0]/num_pitches_to_lefties, 
                            color='b', alpha=.5, s=10, marker = '$L$')
            except ZeroDivisionError:
                pass
            if plot_num > 16:
                plt.gca().text(index, -.1, pitch_type, ha='center', fontsize=8)
        plt.ylim([0, 1])
        plt.xticks([])
        if plot_num > 1:
            plt.gca().yaxis.set_major_formatter( NullFormatter() )

    plt.subplot(5,4,1)
    plt.title('Any Strikes')
    plt.ylabel('Any Balls')

    plt.subplot(5,4,2)
    plt.title('0 Strikes')

    plt.subplot(5,4,3)
    plt.title('1 Strike')

    plt.subplot(5,4,4)
    plt.title('2 Strikes')

    plt.subplot(5,4,5)
    plt.ylabel('0 Balls')

    plt.subplot(5,4,9)
    plt.ylabel('1 Ball')

    plt.subplot(5,4,13)
    plt.ylabel('2 Balls')

    plt.subplot(5,4,17)
    plt.ylabel('3 Balls')
    plt.gca().text(-.25, -.2, 'Increasing velocity -->', ha='left', fontsize=8)
        
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
    app.run(port=33507, debug=True)
