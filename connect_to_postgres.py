# This works on heroku

import os
import psycopg2
import urlparse
import sqlalchemy

urlparse.uses_netloc.append("postgres")
url = urlparse.urlparse(os.environ["DATABASE_URL"])

database=url.path[3:-3],
user=url.username[2:-3],
password=url.password[2:-3],
host=url.hostname[2:-3],
port=url.port
scheme = url.scheme[2:-3]

print '%s://%s:%s@%s:%s/%s' % (scheme, user, password, host, port, database)

engine = sqlalchemy.create_engine('%s://%s:%s@%s:%s/%s' % (scheme, user, password, host, port, database))

conn = engine.connect()

query = "SELECT * FROM pitches LIMIT 10"

df = pd.read_sql(query, engine)

conn.close()

# # This works locally

# import os
# import psycopg2
# import urlparse
# import sqlalchemy
# import pandas as pd

# urlparse.uses_netloc.append("postgres")
# url = urlparse.urlparse(os.environ["DATABASE_URL"])

# database=url.path[1:],
# user=url.username,
# password=url.password,
# host=url.hostname,
# port=url.port

# engine = sqlalchemy.create_engine('postgres://greg@localhost:5432/pitchfx')
# conn = engine.connect()

# query = "SELECT * FROM pitches LIMIT 10"

# df = pd.read_sql(query, engine)

# print df

# conn.close()

# # This works on both

# import os
# import psycopg2
# import urlparse
# import sqlalchemy
# import pandas as pd

# urlparse.uses_netloc.append("postgres")
# url = urlparse.urlparse(os.environ["DATABASE_URL"])

# database=url.path[1:],
# user=url.username,
# password=url.password,
# host=url.hostname,
# port=url.port

# if host == ('localhost',):
#     engine = sqlalchemy.create_engine('postgres://greg@localhost:5432/pitchfx')
# else:
#     engine = sqlalchemy.create_engine('postgresql+psycopg2://%s:%s@%s:%s/%s' % (user, password, host, port, database))

# conn = engine.connect()

# query = "SELECT * FROM pitches LIMIT 10"

# df = pd.read_sql(query, engine)

# print df

# conn.close()