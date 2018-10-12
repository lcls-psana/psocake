import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument
import datetime
from pprint import pprint
import os

class LabelDatabase:

    def __init__(self):
        """ Connect to the database of peakfinding information, and then start a new log of information from this client.
        Arguments:
        kwargs -- peakfinding parameters, host and server name, client name
        """
        print("LabelDatabase init")
        dbname = "labels"
        collectionName = "classifications"
        this_path = os.path.dirname(os.path.realpath(__file__))
        parentDir = os.path.abspath(os.path.join(this_path, os.pardir))
        filename = parentDir + "/scripts/databaseLocation.txt"
        file = open(filename, "r")
        server = file.read()
        self.client = MongoClient('mongodb://%s:27017/'%server)
        self.db = self.client[dbname] # creates a database named "labels"
        self.poster = self.db[collectionName]
        self.printDatabase()

    def postClasses(self, data):
        cursor = self.poster.find_one_and_replace({'classes':{"$exists" : True}},
                                                  data,
                                                  upsert=True)
        print("##### printDatabase:")
        self.printDatabase()

    def post(self, data):
        """ Post data to this database

        Arguments:
        name -- post name
        data_type -- the type of data to be posted, as of now, either labels or classifications
        data -- the data to be posted
        """
        #if(name == None):
        #    dictionary = {"$set":{"%s.%s"%(clientName,data_type):data}}
        #    self.poster.find_one_and_update({clientName:{"$exists":True}},dictionary,upsert = True)
        #else:
        #dictionary = {"$set":{"%s.%s"%(name,data_type):data}}
        #print("dictionary: ", data)
        cursor = self.poster.find_one_and_replace({'$and': [{'user': data["user"]},
                                                 {'exp': data["exp"]},
                                                 {'run': data["run"]},
                                                 {'event': data["event"]}]},
                                                 data,
                                                 upsert=True)
        pprint(cursor)#print("cursor: ", cursor.count())
        #if cursor.count() > 0:
        #    print("$$$$ FOUND ONE")#self.poster.find_one_and_update()
        #else:
        #    print("$$$$ NOOOOO")
        #    self.poster.insert_one(data)#, upsert = True)
        print("##### printDatabase:")
        self.printDatabase()
        #a = self.poster.find_one({"user":"yoon82"})
        #pprint(a)

    def printDatabase(self):
        """ Pretty prints each dictionary stored within the database.
        """
        cursor = self.poster.find({})
        for document in cursor:
            pprint(document)

    def findPost(self, postName):
        try:
            return self.poster.find_one({postName:{"$exists" : True}})
        except:
            return None

    def resetDatabase(self):
        self.poster.delete_many({})

