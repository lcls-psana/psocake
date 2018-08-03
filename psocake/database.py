import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument
import datetime
from pprint import pprint

class LabelDatabase:

    def __init__(self):
        """ Connect to the database of peakfinding information, and then start a new log of information from this client.
        Arguments:
        kwargs -- peakfinding parameters, host and server name, client name
        """
        server = "psanagpu114"#########TODO
        self.client = MongoClient('mongodb://%s:27017/'%server)
        dbname = "Labels"
        self.db = self.client[dbname]
        self.poster = self.db.posts
        self.clientName = datetime.datetime.now().strftime("--%Y-%m-%d--%H:%M:%S")


    def post(self, name, labels):
        if(name == None):
            dictionary = {self.clientName:labels}
            self.theid = self.poster.insert_one(dictionary).inserted_id
            print(self.theid)
        else:
            dictionary = {name:labels}
            self.theid = self.poster.insert_one(dictionary).inserted_id
            print(self.theid)

    def printDatabase(self):
        """ Pretty prints each dictionary stored within the database.
        """
        cursor = self.poster.find({})
        for document in cursor: 
            pprint(document)

    def findPost(self, postName):
        return self.poster.find_one({postName:{"$exists" : True}})

    def resetDatabase(self):
        self.poster.delete_many({})
