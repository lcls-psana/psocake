import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument
import datetime
from pprint import pprint
import os

class LabelDatabase:

    def __init__(self, directory):
        """ Connect to the database of peakfinding information, and then start a new log of information from this client.
        Arguments:
        kwargs -- peakfinding parameters, host and server name, client name
        """
        this_path = os.path.dirname(os.path.realpath(__file__))
        parentDir = os.path.abspath(os.path.join(this_path, os.pardir))
        filename = parentDir + "/scripts/databaseLocation.txt"
        file = open(filename, "r")
        server = file.read()
        self.client = MongoClient('mongodb://%s:27017/'%server)
        dbname = "Labels"
        self.db = self.client[dbname]
        self.poster = self.db.posts
        self.clientName = datetime.datetime.now().strftime("--%Y-%m-%d--%H:%M:%S")


    def post(self, name, data_type, data):
        """ Post data to this database

        Arguments:
        name -- post name
        data_type -- the type of data to be posted, as of now, either labels or classifications
        data -- the data to be posted
        """
        if(name == None):
            dictionary = {"$set":{"%s.%s"%(clientName,data_type):data}}
            self.poster.find_one_and_update({clientName:{"$exists":True}},dictionary,upsert = True) 
        else:
            dictionary = {"$set":{"%s.%s"%(name,data_type):data}}
            self.poster.find_one_and_update({name:{"$exists":True}},dictionary,upsert = True) 
        self.printDatabase()

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
