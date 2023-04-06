from pathlib import Path

def exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path

def mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
