from tqdm import tqdm
from .parser import ChemblParser
from utils.dbutils import DBConnection
from utils.ConfigParser import ConfigParser


class ChembltoMongo:
    def __init__(self, config):
        self.config = config
        self.db = DBConnection(self.config.get("db_name"), self.config.get("col_name"), config=self.config)
    def start(self, parser, use_progress_bar=True):
        self.db.add_index("chembl_id")
        for data in tqdm(parser.start(), total=parser.get_batch_num(), disable=not use_progress_bar):
            self.db.insert(data, accelerate=True, buffer_size=10000)


if __name__ == "__main__":
    cfg = "conf/drugkb.config"
    config = ConfigParser(cfg)
    config.set_section("chembl")
    to_mongo = ChembltoMongo(config)
    parser = ChemblParser(config)
    ChembltoMongo(config).start(parser)