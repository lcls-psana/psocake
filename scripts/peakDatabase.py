import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument
import datetime
from pprint import pprint

class PeakDatabase:

    def __init__(self, **kwargs):
        """ Connect to the database of peakfinding information, and then start a new log of information from this client.

        Arguments:
        kwargs -- peakfinding parameters, host and server name, client name
        """
        server = kwargs["server"]
        self.client = MongoClient('mongodb://%s:27017/'%server)
        dbname = kwargs["dbname"]
        self.db = self.client[dbname]
        self.poster = self.db.posts
        self.clientName = kwargs["name"] + datetime.datetime.now().strftime("--%Y-%m-%d--%H:%M:%S")
        dictionary = {self.clientName:{"parameters":{"npxmin" : kwargs["npix_min"],
                                                    "npxmax" : kwargs["npix_max"],
                                                    "amaxthr" : kwargs["amax_thr"],
                                                    "atotthr" : kwargs["atot_thr"],
                                                    "sonmin" : kwargs["son_min"]}}}
        self.theid = self.poster.insert_one(dictionary).inserted_id

    def addExpRunEventPeaks(self, **kwargs):
        """ Adds the number of peaks found for an event in an experment run.

        Arguments:
        kwargs -- Exp, Run, Event, Peaks
        """
        header1 = self.clientName + "." + kwargs["Exp"] + "." + kwargs["RunNum"] + "." + kwargs["Event"] + "." +"Peaks"
        header2 = self.clientName + "." + kwargs["Exp"] + "." + kwargs["RunNum"] + "." + kwargs["Event"] + "." +"Labels"
        self.poster.find_one_and_update({u'_id': self.theid},
                                        {"$set":{header1: kwargs["Peaks"], header2: kwargs["Labels"]}},
                                        upsert = True,
                                        return_document=ReturnDocument.AFTER)

    def addPeaknetModel(self):
        pass

    def returnPeaknetModel(self):
        pass

    def printDatabase(self):
        """ Pretty prints each dictionary stored within the database.
        """
        cursor = self.poster.find({})
        for document in cursor: 
            pprint(document)

    def addPeaksAndHits(self, peaks, hits):
        peakHeader = "Peaks"
        hitHeader = "Hits"
        try:
            peaks = peaks + self.poster.find_one({"Peaks":{"$exists":True}})["Peaks"]
            hits = hits + self.poster.find_one({"Hits":{"$exists":True}})["Hits"]
        except TypeError:
            pass
        self.poster.find_one_and_update({"Peaks":{"$exists":True}},
                                        {"$set":{peakHeader: peaks}},
                                        upsert = True,
                                        return_document=ReturnDocument.AFTER)
        self.poster.find_one_and_update({"Hits":{"$exists":True}},
                                        {"$set":{hitHeader: hits}},
                                        upsert = True,
                                        return_document=ReturnDocument.AFTER)

    def returnPeaks(self):
        try:
            peaks = self.poster.find_one({"Peaks":{"$exists":True}})["Peaks"]
        except TypeError:
            return 0
        return peaks

    def returnHits(self):
        try:
            hits = self.poster.find_one({"Hits":{"$exists":True}})["Hits"]
        except TypeError:
            return 0
        return hits

    def resetDatabase(self):
        self.poster.delete_many({})
        self.printDatabase()
