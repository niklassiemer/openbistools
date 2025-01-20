import metadata_ZeissLeo as ZeissLeo
from keys.keys_SEMDict import keys_SEMDict
from pprint import pprint


if __name__ == '__main__':
    tif_path = '../data/Zeiss_LEO/ZEISS_LEO_pano000.tif'
    metadata = ZeissLeo.get_metadata(tif_path, keys_SEMDict)
    print('-' * 10 + 'metadata' + '-' * 10)
    pprint(metadata)
