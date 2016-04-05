# This works on heroku

import os
import psycopg2
import urlparse
import sqlalchemy

urlparse.uses_netloc.append("postgres")
url = urlparse.urlparse(os.environ["DATABASE_URL"])

database=url.path,
user=url.username,
password=url.password,
host=url.hostname,
port=url.port
scheme = url.scheme

print "database: ", database[0], type(database)
print "user: ", user[0], type(user)
print "password: ", password, type(password)
print "host: ", host[0], type(host)
print "port: ", post[0], type(post)
print "scheme: ", scheme[0], type(scheme)

print '%s://%s:%s@%s:%s/%s' % (scheme[0], user[0], password[0], host[0], port[0], database[0])

engine = sqlalchemy.create_engine('%s://%s:%s@%s:%s/%s' % (scheme[0], user[0], password[0], host[0], port[0], database[0]))

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