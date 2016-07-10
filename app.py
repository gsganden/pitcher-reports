from flask import Flask, render_template, request, redirect
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import sqlalchemy
import seaborn as sns
from sklearn.mixture import GMM
from matplotlib.ticker import NullFormatter
import matplotlib.patches as mpatches
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

sns.set_context('notebook')

pitch_type_dict = dict(FA='fastball',
                       FF='four-seam fastball',
                       SI='sinker',
                       FC='fastball (cutter)',
                       SL='slider',
                       CH='changeup',
                       CU='curveball',
                       KC='knuckle-curve',
                       KN='knuckleball',
                       EP='eephus')

@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        pitcher = request.form['pitcher']
        season = int(request.form['season'])
        # try:
        data, pitch_types = get_data(pitcher.lower(), season)
        # except:
            # return render_template('error.html')
        return render_template('results.html',
                               repertoire_plot=plot_repertoire(data, pitch_types),
                               selection_plot=plot_selection(data, pitch_types),
                               location_plot=plot_location(data, pitch_types),
                               pitcher=pitcher.title(),
                               season=season)


def get_data(pitcher, season):
    query = '''
            SELECT *
            FROM pitches_app
            WHERE pitcher = '%s'
                AND year = %s
            ''' % (pitcher, season)

    data = pd.read_sql(query, engine)

    pitch_types = sorted(list(data['pitch_type'].unique()))
    # Make copy to avoid altering list while iterating over it
    filtered_pitch_types = pitch_types[:]

    # Get rid of very infrequent pitch types, which are mostly bad data
    for pitch_type in pitch_types:
        if float(data[data['pitch_type'] == pitch_type].shape[0])\
                / data.shape[0] < .02:
            filtered_pitch_types.remove(pitch_type)

    pitch_types = list(data[data['pitch_type'].isin(filtered_pitch_types)]
                       .groupby('pitch_type')
                       .agg({'start_speed': [np.size, np.mean]})
                       ['start_speed']
                       .sort_values(by='mean')
                       .index)

    return data, pitch_types


def plot_repertoire(data, pitch_types):
    gaussians = []
    gmms = []

    for pitch_type in pitch_types:
        gmm = GMM(covariance_type='full')

        sub_data = data[data['pitch_type'] == pitch_type][['pfx_x',
                                                           'pfx_z',
                                                           'start_speed']]
        gmm.fit(sub_data)

        x = np.arange(-20, 20, 0.25)
        y = np.arange(-20, 20, 0.25)
        X, Y = np.meshgrid(x, y)

        gmms.append(gmm)

        gaussians\
            .append(plt.mlab
                       .bivariate_normal(X,
                                         Y,
                                         sigmax=np.sqrt(gmm
                                                        ._get_covars()
                                                        [0][0][0]),
                                         sigmay=np.sqrt(gmm
                                                        ._get_covars()
                                                        [0][1][1]),
                                         sigmaxy=gmm._get_covars()[0][0][1],
                                         mux=gmm.means_[0][0],
                                         muy=gmm.means_[0][1]))

    plt.figure(figsize=(8, 6))
    plt.scatter(data['pfx_x'], data['pfx_z'], c=data['start_speed'],
                alpha=.3, cmap='inferno', norm=Normalize(70, 100), s=10)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.yticks([-10, 0, 10])
    plt.xticks([-10, 0, 10])
    plt.ylabel('Vertical Break (Inches)')
    plt.xlabel('Horizontal Break (Inches)   ')
    plt.colorbar().set_label('Velocity')
    ax = plt.gca()
    ax.text(.61,
            .98,
            ''.join([pitch_type + ': ' + pitch_type_dict[pitch_type] + '\n'
                     for pitch_type in pitch_types])[:-1],
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)\
        .set_bbox(dict(color='w', alpha=0.3, edgecolor='k'))

    for index in xrange(len(gaussians)):
        plt.contour(X, Y, gaussians[index], 3, colors='k', alpha=.3)
        ax.text(gmms[index].means_[0][0],
                gmms[index].means_[0][1],
                pitch_types[index],
                ha='center',
                va='center',
                color='k',
                size=10,
                backgroundcolor='w')\
            .set_bbox(dict(color='w', alpha=0.3, edgecolor='k'))

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


def plot_selection(data, pitch_types):
    plt.figure(figsize=(8, 5))

    for plot_num in range(1, 21):
        plt.subplot(4, 5, plot_num)
        num_strikes = ((plot_num - 1) // 5) - 1
        num_balls = ((plot_num - 1) % 5) - 1
        pitch_data = data.copy()
        if num_balls > -1:
            pitch_data = pitch_data[pitch_data['balls'] == num_balls]
        if num_strikes > -1:
            pitch_data = pitch_data[pitch_data['strikes'] == num_strikes]
        num_pitches_to_righties = float(pitch_data[pitch_data['stand'] == 'R']
                                        .shape[0])
        num_pitches_to_lefties = float(pitch_data[pitch_data['stand'] == 'L']
                                       .shape[0])
        for index, pitch_type in enumerate(pitch_types):
            filter_pitch_data = pitch_data[pitch_data['pitch_type'] ==
                                           pitch_type]
            try:
                plt.scatter(index,
                            filter_pitch_data[filter_pitch_data['stand'] ==
                                              'R']
                            .shape[0]/num_pitches_to_righties,
                            color='r',
                            alpha=.5)
            except ZeroDivisionError:
                pass
            try:
                plt.scatter(index,
                            filter_pitch_data[filter_pitch_data['stand'] ==
                                              'L']
                            .shape[0]/num_pitches_to_lefties,
                            color='b',
                            alpha=.5)
            except ZeroDivisionError:
                pass
            if plot_num > 15:
                plt.gca().text(index, -.15, pitch_type, ha='center', fontsize=8)
        plt.ylim([0, 1])
        plt.xticks([])
        if plot_num != 1:
            plt.gca().yaxis.set_major_formatter(NullFormatter())

    plt.subplot(4, 5, 1)
    plt.title('Any Balls')
    plt.ylabel('Any Strikes')

    plt.subplot(4, 5, 2)
    plt.title('0 Balls')

    plt.subplot(4, 5, 3)
    plt.title('1 Ball')

    plt.subplot(4, 5, 4)
    plt.title('2 Balls')

    plt.subplot(4, 5, 5)
    plt.title('3 Balls')
    red_patch = mpatches.Patch(color='red', label='Righty batter')
    blue_patch = mpatches.Patch(color='blue', label='Lefty batter')
    plt.legend(handles=[red_patch, blue_patch])

    plt.subplot(4, 5, 6)
    plt.ylabel('0 Strikes')

    plt.subplot(4, 5, 11)
    plt.ylabel('1 Strike')

    plt.subplot(4, 5, 16)
    plt.ylabel('2 Strikes')

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


def plot_location(data, pitch_types):
    plt.figure(figsize=(3, 2*len(pitch_types)))

    righty_data = data[data['stand'] == 'R']
    lefty_data = data[data['stand'] == 'L']
    pitch_type_num = -1
    for plot_num in range(1, 2 * len(pitch_types) + 1):
        plot_index = plot_num - 1
        if plot_index % 2 == 0:  # new pitch type
            pitch_type_num += 1
            righty_pitch_data = righty_data[righty_data['pitch_type'] ==
                                            pitch_types[pitch_type_num]]
            lefty_pitch_data = lefty_data[lefty_data['pitch_type'] ==
                                          pitch_types[pitch_type_num]]
        plt.subplot(len(pitch_types), 2, plot_num)
        plt.plot([-.7083, .7083, .7083, -.7083, -.7083],
                 [0, 0, 1, 1, 0])  # Strike zone
        plt.ylim([-1.5, 2])
        plt.xlim([-3, 3])
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(NullFormatter())
        if plot_num % 2 == 1:
            plt.scatter(righty_pitch_data['px'],
                        righty_pitch_data['pz'],
                        c='r', alpha=.2, s=10)
            plt.ylabel(pitch_types[pitch_type_num], rotation=0)
            ax.yaxis.labelpad = 15
        else:
            plt.scatter(lefty_pitch_data['px'],
                        lefty_pitch_data['pz'],
                        c='b', alpha=.2, s=10)
        if plot_index == 0:
            plt.title('Righty batters')
        elif plot_index == 1:
            plt.title('Lefty batters')

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
