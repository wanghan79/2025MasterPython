import json
import logging
import os.path
import time
from collections import deque

import pymongo
from bson.objectid import ObjectId
from tqdm import tqdm

from ConfigParser import ConfigParser

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class DBConnection:
    """
    The DBConnection is a class for data store, now only insert function supported.

    Parameters:
        - db_name (str): The name of MongoDB database
        - db_url (str, optional): Database url, the default value is "127.0.0.1".
        - port (int, optional): Database port, the default value is 27017
        - empty_check (bool, optional): Determine whether to clear the corresponding collection in the database before inserting data. The default is False.
        - user_name(str, optional): MongoDB username. This parameter is optional. By default, the username and password mode are not used
        - password(str, optional): MongoDB password
        - cfg_file(str,optional): use config file property

    Methods:
        - assure_empty(): Empty the specified collection in the database. If empty_check is set to True, this method is called before the data is automatically inserted.
        - insert(data: iter, buffer_size=10000, accelerate=False, counter=False): Inserts data from the specified csv file into the corresponding collection.
        - clear_collection(): Clear collection
    """

    def __init__(self, db_name: str, collection_name: str, host: str = "127.0.0.1", port: int = 27017,
                 username: str = None, password: str = None, empty_check=True, config=None):
        if config is None:
            pass
        else:
            host = config.get('host', 'dbserver')
            port = int(config.get('port', 'dbserver').strip())
            username = config.get('user', 'dbserver')
            password = config.get('password', 'dbserver')

        self.client = pymongo.MongoClient(host, port) \
            if username is None and password is None \
            else pymongo.MongoClient(host, port, username=username, password=password)
        self.db = self.client[db_name]
        self.collection_name = collection_name
        self.collection = self.db[collection_name]
        self.collection_structure = {}
        self.str_max_len = 40
        self.unexpected_type = set()
        self.__assure_empty(empty_check)

    def insert(self, data: iter, buffer_size=10000, accelerate=False, counter=False) -> None:
        """
            insert a well-organized dict items, so generator is recommended.
            :param data: iterable data
            :param buffer_size: The size of the buffer when data is inserted. The default is 10000
            :param accelerate: enable multi-thread insert. Note that the insert action is unordered.
            :param counter: enable data count when is set True
        """
        logging.debug(f"Collection {self.collection_name} start conversion.")
        beginning_time = time.time()
        data_counter = 0

        buffer = deque()
        for item in data:
            buffer.append(item)
            if len(buffer) >= buffer_size:
                requests = [pymongo.InsertOne(doc) for doc in buffer]
                self.collection.bulk_write(requests, ordered=accelerate)
                if counter:
                    data_counter += buffer_size
                buffer.clear()
        # Ensure that all data in the buffer is retrieved
        if buffer:
            requests = [pymongo.InsertOne(doc) for doc in buffer]
            self.collection.bulk_write(requests, ordered=accelerate)
            if counter:
                data_counter += len(buffer)
            buffer.clear()
        if counter:
            logging.debug(
                f"Collection {self.collection_name} conversion complete，Insert {data_counter} bar documents, time elapsed {time.time() - beginning_time:.2f}s")
        else:
            logging.debug(
                f"Collection {self.collection_name} conversion complete, time elapsed {time.time() - beginning_time:.2f}s")

    def __assure_empty(self, dbCheck: bool) -> None:
        # If dbCheck is set to True, this method is called before the data is automatically inserted.
        if dbCheck:
            if self.collection.find_one({}) is None:
                logging.debug("Collection is empty!")
            else:
                logging.error("Collection not empty.")
                self.collection.drop()
    def clear_collection(self):
        self.collection.drop()

    def search_record(self, query: dict) -> dict:
        return self.collection.find_one(query)

    def print_struct_fields(self, collection_name: str, str_max_len=40, save_as_file=False, file_path="./"):
        """
        automatic travel all doc to find the most detailed fields.
        :param collection_name: mongo collection name, can give any collection name in this db.
        :param str_max_len: max length of string type field
        :param save_as_file: bool, whether to save fields info as json file
        """
        self.str_max_len = str_max_len
        collection = self.db[collection_name]
        for doc in tqdm(collection.find(), desc=collection_name):
            self.field_dfs(doc, self.collection_structure)
        # TODO add fields analyze function
        if save_as_file:
            with open(os.path.join(file_path, f"{collection_name}.json"), "w") as f:
                json.dump(self.collection_structure, f, indent=4)
        else:
            print(json.dumps(self.collection_structure, indent=4))

    def field_dfs(self, data, structure):
        if isinstance(data, dict):
            for key, value in data.items():
                if not structure.get(key, None) or isinstance(structure[key], (dict, list)):
                    if isinstance(value, ObjectId):
                        continue
                    if value is None or isinstance(value, (int, float, str, bool)):
                        structure[key] = value[:self.str_max_len] if isinstance(value, str) else value
                    elif isinstance(value, list):
                        structure[key] = [None]
                        structure[key] = self.field_dfs(value, structure[key])
                    elif isinstance(value, dict):
                        structure[key] = dict()
                        structure[key] = self.field_dfs(value, structure[key])
                    else:
                        print(f"unexpected type in sub judge, type: {type(value)}")
                        self.unexpected_type.add(type(value))

        elif isinstance(data, list):
            for value in data:
                if not structure[0] or isinstance(structure[0], (dict, list)):
                    if isinstance(value, ObjectId):
                        continue
                    if value is None or isinstance(value, (int, float, str, bool)):
                        structure[0] = value[:self.str_max_len] if isinstance(value, str) else value
                    elif isinstance(value, list):
                        structure[0] = [None]
                        structure[0] = self.field_dfs(value, structure[0])
                    elif isinstance(value, dict):
                        structure[0] = dict()
                        structure[0] = self.field_dfs(value, structure[0])
                    else:
                        print(f"unexpected type in sub judge, type: {type(value)}")
                        self.unexpected_type.add(type(value))

        else:
            print(f"unexpected type in main judge, type: {type(data)}")
            self.unexpected_type.add(type(data))

        return structure

    def close(self):
        self.client.close()

    def get_iter(self, collection_name: str):
        return self.db[collection_name].find()

    def set_col(self, col):
        self.collection = self.db[col]

    def find_one(self, query, target_col=""):
        if not target_col:
            return self.collection.find_one(query)
        else:
            return self.db[target_col].find_one(query)

    def find_many(self, query, target_col=""):
        if not target_col:
            return self.collection.find(query)
        else:
            return self.db[target_col].find(query)

    def update_one(self, query, update, target_col=""):
        if not target_col:
            return self.collection.update_one(query, update)
        else:
            return self.db[target_col].update_one(query, update)

    def delete_one(self, query, target_col=""):
        if not target_col:
            return self.collection.delete_one(query)
        else:
            return self.db[target_col].delete_one(query)

    def add_index(self, name, direction=pymongo.ASCENDING, is_unique=True):
        self.collection.create_index([(name, direction)], unique=is_unique)

if __name__ == "__main__":
    # The following is a usage example
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d %I:%M:%S')
    # insert data into collection "offsides" in database "drugdb" whose server at "59.168.1.100"
    # disable collection empty check
    drugDB = DBConnection("drugdb", "offsides", "59.168.1.100", username="root", password="pw123", empty_check=False)
    # If you use config
    cfg_file = "../conf/drugkb_test.config"
    config = ConfigParser(cfg_file)
    drugDB = DBConnection("drugdb", "offsides", config=config)
    """
    Recommend Usage
    # large data case, you should pass a generator
    def generate_object():
        for row in db_reader:
            # wrapped as a dict object (IMPORTANT)
            drug = dict(zip(header, [row[i] for i in field_allowed]))
            yield drug

    db.insert(generate_object, accelerate=True)

    # normal size data, pass a list is ok
    db.insert([dict1,dict2,...], accelerate=True)
    """