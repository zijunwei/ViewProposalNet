# Shrinked and adapted from z_utils.py
import os
import shutil
import sys
from datetime import datetime


def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')[:-7]


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)


class save_places:
    def __init__(self, dir_name):
        self.save_dir = get_dir(dir_name)

        self.model_save_dir = get_dir(os.path.join(self.save_dir, 'models'))
        self.summary_save_dir = get_dir(os.path.join(self.save_dir, 'summaries'))
        self.image_save_dir = get_dir(os.path.join(self.save_dir, 'images'))
        self.log_save_dir = get_dir(os.path.join(self.save_dir, 'logs'))

    def clear_save_name(self):
        """
        Clears all saved content for SAVE_NAME.
        """
        clear_dir(self.model_save_dir)
        clear_dir(self.summary_save_dir)
        clear_dir(self.log_save_dir)
        clear_dir(self.image_save_dir)
        print ('Clear stuff in {}'.format(os.path.join(self.save_dir)))

