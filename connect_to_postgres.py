import os
import psycopg2
import urlparse

# urlparse.uses_netloc.append("postgres")
# url = urlparse.urlparse(os.environ["DATABASE_URL"])

# conn = psycopg2.connect(
#     database=url.path[1:],
#     user=url.username,
#     password=url.password,
#     host=url.hostname,
#     port=url.port
# )

# cur = conn.cursor()

# cur.execute("SELECT * FROM pitches LIMIT 10;")
# print cur.fetchone()

# cur.close()
# conn.close()

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

query = "SELECT * FROM games LIMIT 10;"

query_result = run_query(query)

print query_result