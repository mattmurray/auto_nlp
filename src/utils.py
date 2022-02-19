import logging
import pathlib
from datetime import datetime
import requests
import yaml
from tqdm import tqdm

def get_timestamp(include_seconds=True):
    """
    Creates a timestamp string based on the current time.
    :param include_seconds:     Boolean, default is True. Includes seconds in the string.
                                If False, will include year, month, day, hour, minutes.
    :return:                    String, of current timestamp
    """

    if include_seconds:
        timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
    else:
        timestamp = datetime.today().strftime('%Y%m%d%H%M')

    return timestamp


def create_directory(parent_directory, name=None):
    """
    :param parent_directory:    String, parent directory where the new directory will be created.
    :param name:                Optional, if None then current timestamp is created.
    :return:                    String, the location of the new directory created.
    """

    if name is None:
        name = get_timestamp(True)

    # the new path to create
    new_path = pathlib.Path(parent_directory) / name

    # checks if directory already exists and raises exception
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path


def download_file(url, out_path):
    local_filename = url.split('/')[-1]
    out_file = pathlib.Path(out_path / local_filename)

    with requests.get(url, stream=True) as r:
        total = int(r.headers.get('content-length', 0))
        with open(out_file, 'wb') as file, tqdm(
                desc=local_filename,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in r.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    return str(out_file)


def load_yaml(path):
    with open(path, "r") as stream:
        try:
            yaml_file = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            raise Exception(exc)

    return yaml_file


def create_logger(name, file_path, level='debug', file_level='debug', console_level='error'):
    logger = logging.getLogger(name)

    if level == 'error':
        logger.setLevel(logging.ERROR)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.DEBUG)

    # create file handler to log messages
    fh = logging.FileHandler(file_path)

    if file_level == 'error':
        fh.setLevel(logging.ERROR)
    elif file_level == 'info':
        fh.setLevel(logging.INFO)
    elif file_level == 'warning':
        fh.setLevel(logging.WARNING)
    else:
        fh.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()

    if console_level == 'debug':
        ch.setLevel(logging.DEBUG)
    elif console_level == 'info':
        ch.setLevel(logging.INFO)
    elif console_level == 'warning':
        ch.setLevel(logging.WARNING)
    else:
        ch.setLevel(logging.ERROR)

    # log formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

