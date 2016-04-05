# This works on heroku

import os
import psycopg2
import urlparse
import sqlalchemy
import pandas as pd

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

query = "SELECT * FROM pitches LIMIT 10"

df = pd.read_sql(query, engine)

print df

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