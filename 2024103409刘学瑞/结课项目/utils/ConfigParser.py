import configparser
import os.path
class ConfigParser:

    @staticmethod
    def get_config(path):
        '''
        Read the configuration file
        @return:config object
        '''
        config = configparser.ConfigParser()
        if not os.path.exists(path):
            return None
        else:
            cfgfile = path
            config.read(cfgfile)
            if len(config) == 1:
                return None
            else:
                return config

    def __init__(self, path, section=None):
        self.config = self.get_config(path)
        self.section = section
        if self.config is None:
            raise Exception("Config file not found")

    def set_section(self, section):
        self.section = section

    def get(self, option, section=None):
        if section is None:
            section = self.section
        if section is None:
            raise Exception("Config section not found")
        return self.config.get(section, option)


if __name__ == '__main__':
    ConfigParser.get_config(r'../conf/drugkb.config')