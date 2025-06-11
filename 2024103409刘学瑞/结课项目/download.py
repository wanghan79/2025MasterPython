import os
import shutil
import threading
from pathlib import Path

from DownloadKit import DownloadKit

from utils.ConfigParser import ConfigParser
import tarfile


class ChEMBLDownloader:
    def __init__(self, config):
        self.config = config
        self.downloadKit = DownloadKit()

    def extract_tar_gz(self, tar_path: Path):
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(tar_path.parent)

    def start(self):
        urls = [self.config.get(f'source_url_{i}') for i in range(1, int(self.config.get('source_url_num')) + 1)]
        data_paths = [Path(self.config.get(f'data_path_{i}')) for i in range(1, int(self.config.get('data_path_num')) + 1)]
        # overwrite the file if exists
        threads = [threading.Thread(target= lambda url, save_path: self.downloadKit.add(url, save_path, file_exists='o').wait(show=True), args=(url, save_path))
                   for url, save_path in zip(urls, data_paths)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for data_path in data_paths:
            for f in list(data_path.rglob('*')):
                self.extract_tar_gz(f)
                chembl_version = "_".join(f.stem.split("_")[:-1])
                chembl_mysql = f"{chembl_version}_mysql"
                shutil.move(f.parent / chembl_version / chembl_mysql / f"{chembl_mysql}.dmp", f.parent)
                # clean tmp file
                shutil.rmtree(f.parent / chembl_version)
                f.unlink()




if __name__ == '__main__':
    cfg = "conf/drugkb.config"
    config = ConfigParser(cfg)
    config.set_section("chembl")
    ChEMBLDownloader(config).start()