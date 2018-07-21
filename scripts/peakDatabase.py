import pymongo
from pymongo import MongoClient
from pymongo import ReturnDocument

class PeakDatabase:

    def __init__(self, **kwargs):
        server = kwargs["server"]
        self.client = MongoClient('mongodb://%s:27017/'%server)
        self.db = self.client['PeakFindingDatabase']
        self.poster = self.db.posts
        self.clientName = kwargs["name"]
        dictionary = {self.clientName:{"parameters":{"npxmin" : kwargs["npix_min"],
                                                    "npxmax" : kwargs["npix_max"],
                                                    "amaxthr" : kwargs["amax_thr"],
                                                    "atotthr" : kwargs["atot_thr"],
                                                    "sonmin" : kwargs["son_min"]}}}
        self.theid = self.poster.insert_one(dictionary).inserted_id

    def addExpRunEventPeaks(self, **kwargs):
        header = self.clientName + "." + kwargs["Exp"] + "." + kwargs["RunNum"] + "." + kwargs["Event"]
        self.poster.find_one_and_update({u'_id': self.theid},
                                        {"$set":{header: kwargs["Peaks"]}},
                                        upsert = True,
                                        return_document=ReturnDocument.AFTER)

