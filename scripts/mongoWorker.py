from pymongo import MongoClient

host = 'psanagpu114'
port = 27017 # default

client = MongoClient(host, port)
db = client['test-database']
collection = db['test-collection']
posts = db.posts
posts
import pprint
print "trying to fetch from server..."
for post in posts.find():
    pprint.pprint(post)
