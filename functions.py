import os


def search_folder(dir_location, fileextension):
    results = [os.path.join(root, name)
               for root, dirs, files in os.walk(dir_location)
               for name in files if name.endswith(fileextension)]

    return results
