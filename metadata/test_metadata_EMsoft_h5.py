from pprint import pprint
import metadata_EMsoft_h5

print('=============================== EMsoft hdf5 file Metadata ================================================')
metadata = metadata_EMsoft_h5.get_metadata('../../data/EMsoft/EBSDout_Contcar1.h5')
pprint(metadata)
