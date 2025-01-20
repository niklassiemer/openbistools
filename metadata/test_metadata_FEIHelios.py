import metadata_FEIHelios as FEIHelios
from keys.keys_SEMDict import keys_SEMDict
from pprint import pprint


if __name__ == '__main__':
    tif_path = '../data/FEI_Helios/FEI_HELIOS_ebsd_slip_events_pos5_005.tif'
    # tif_path = '../data/FEI_Helios/FEI_Helios_Pillar22_1_22.tif'
    metadata = FEIHelios.get_metadata(tif_path, keys_SEMDict)
    print('-' * 10 + 'metadata' + '-' * 10)
    pprint(metadata)
