import metadata_TescanClara as TescanClara
from keys.keys_SEMDict import keys_SEMDict
from pprint import pprint


if __name__ == '__main__':
    tif_path = '../data/TescanClara/TEST.tif'
    metadata = TescanClara.get_metadata(tif_path, keys_SEMDict)
    print('-' * 10 + 'metadata' + '-' * 10)
    pprint(metadata)
