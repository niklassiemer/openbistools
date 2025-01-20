from pprint import pprint

import metadata_EdaxTeam


print('=============================== SPD Metadata ================================================')
spd_header = metadata_EdaxTeam.get_metadata_spd('../../data/EDAX_TEAM/map20230717084300668_0.spd')
pprint(spd_header)
print('=============================================================================================')
print('=============================================================================================')
print('')


print('=============================== IPR Metadata ================================================')
ipr_header = metadata_EdaxTeam.get_metadata_ipr('../../data/EDAX_TEAM/map20230605111102605_0_ZafWt_Al_K.ipr')
pprint(ipr_header)
print('=============================================================================================')
print('=============================================================================================')
print('')

print('=============================== SPC Metadata ================================================')
spc_header = metadata_EdaxTeam.get_metadata_spc('../../data/EDAX_TEAM/map20230717084300668_0.spc')
pprint(spc_header)
