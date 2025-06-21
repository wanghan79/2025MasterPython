import sys
from test_tube import HyperOptArgumentParser

class ProtBertPPIArgParser(HyperOptArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def error(self, message):
        self.print_help(sys.stderr)
        sys.stderr.write('\nError: %s\n' % message)
        sys.exit(2)
