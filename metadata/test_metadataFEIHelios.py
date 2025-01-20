import metadata_FEIHelios as helios

metadata = helios.get_metadata('../../data/FEI_Helios/FEI_HELIOS_ebsd_slip_events_pos5_005.tif')
#metadata = helios.get_metadata('../../data/FEI_Helios/FEI_Helios_Pillar22_1_22.tif')
print(metadata)