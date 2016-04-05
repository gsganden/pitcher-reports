from flask import Flask, render_template, request, redirect
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import psycopg2
import urlparse
import sqlalchemy

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
    data = pd.read_sql(query, engine)
    
    conn.close()

    return data

def make_figure(df):
    fig = plt.figure(figsize=(10,8))

    plt.scatter(df['pfx_x'], df['pfx_z'], c=df['start_speed'], \
                    alpha=.3, cmap='inferno', norm = Normalize(70, 100))
    plt.colorbar().set_label('Velocity')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.xlabel('Horizontal movement')
    plt.ylabel('Vertical movement')
    plt.show()

def main(pitcher_name):
    make_figure(get_data(pitcher_name))

print main('Jon Lester')