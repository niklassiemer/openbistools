from pprint import pprint

import metadata_EdaxTeam_zip


print('=============================== Edax zip file Metadata ================================================')
metadata = metadata_EdaxTeam_zip.get_metadata('../../data/EDAX_TEAM/C14_Sample.zip')
pprint(metadata)
