"""Module providing utlities for benchmark"""

import requests


def download(url, path):
    """Download a file from url and save it to path."""
    response = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)
