
import glob, json
from typing import TypeVar
# START MONGO IN MAC: mongod --config /usr/local/etc/mongod.conf
# START MONGO IN UBUNTU: sudo systemctl start mongod
# https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/
from pymongo import MongoClient
MongoCollection = TypeVar("MongoCollection")

COLLECTION_NAME = f"bionet_intavia"
DB_NAME = "biographies"

def update_intavia_mongo():

    INTAVIA_JSON_BASEPATH = "flask_app/backend_data/intavia_json"

    client = MongoClient("mongodb://localhost:27017/")
    
    db = client[DB_NAME]
    bionet_collection = db[COLLECTION_NAME]

    override_db = input(f"Database already exists. Are you sure you want to override it [yes,NO]?  ") 
    if override_db.lower() == "yes":
        print(f"Replacing it with the new data from the Filesystem ...")
        db.drop_collection(bionet_collection)
        # Update with InTaVia Info
        tot = 0
        for src_path in glob.glob(f"{INTAVIA_JSON_BASEPATH}/*"):
            for ix, filepath in enumerate(glob.glob(f"{src_path}/*.json")):
                print(f"{filepath} [{ix}] [{tot}]")
                intavia_obj = json.load(open(filepath))
                bionet_collection.insert_one(intavia_obj)
                tot += 1
    
    print(f"Final MongoDB has {bionet_collection.count_documents({})} items in Collection: {COLLECTION_NAME}. DataBase: {DB_NAME}")

if __name__ == "__main__":
    update_intavia_mongo()