import os.path


def get_cache_files(file: str, *extensions: str):
    path = os.path.realpath(file)
    dir_path = os.path.dirname(path)
    name = os.path.basename(path).split(".", 1)[0]
    return [os.path.join(dir_path, f"{name}.{ext.strip('.')}") for ext in extensions]


def is_cache_recent(file: str, *caches: str):
    if not all(os.path.exists(cache) for cache in caches):
        return False
    mod_time = os.stat(file).st_mtime
    if not all(os.stat(cache).st_mtime > mod_time for cache in caches):
        return False
    return True
