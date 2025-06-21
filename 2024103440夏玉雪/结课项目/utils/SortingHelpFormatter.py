import argparse
from operator import attrgetter

class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super().add_arguments(actions)
