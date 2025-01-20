import h5py
from datetime import datetime

# Uppercase fields correspond to openBIS propertyTypes codes and propertyTypes dataTypes
# The keys of the mappings are those used in EMsoft

screen_pattern_metadata_mapping = {
    'masterfile': ['MASTERPATTERN_FILENAME', 'VARCHAR'],
    'anglefile': ['EULERANGLE_FILENAME', 'VARCHAR'],
    'anglefiletype': ['EULERANGLE_TYPE', 'VARCHAR'],
    'applyDeformation': ['APPLY_LATTICE_DEFORMATION', 'BOOLEAN'],
    'eulerconvention': ['EULERANGLE_CONVENTION', 'CONTROLLEDVOCABULARY'],
    'includebackground': ['INCLUDE_BACKGROUND', 'BOOLEAN'],
    'energymax': ['EBSDIM_ENERGYRANGE_MAX', 'REAL'],
    'energymin': ['EBSDIM_ENERGYRANGE_MIN', 'REAL'],
    'beamcurrent': ['BEAM_CURRENT_NA', 'REAL'],
    'L': ['EBSD_L', 'REAL'],
    'thetac': ['THETA_CAMERA', 'REAL'],
    'delta': ['CCD_PIXEL_SIZE', 'REAL'],
    'numsx': ['CCD_PIXEL_X', 'INTEGER'],
    'numsy': ['CCD_PIXEL_Y', 'INTEGER'],
    'xpc': ['EBSD_PATTERN_CENTRE_X', 'REAL'],
    'ypc': ['EBSD_PATTERN_CENTRE_Y', 'REAL'],
    'alphaBD': ['ALPHA_BARREL_DISTORTION', 'REAL'],
    'bitdepth': ['BIT_DEPTH', 'VARCHAR'], # Extracted as VARCHAR than converted to INTEGER
    'dwelltime': ['DWELL_TIME', 'REAL'],
    'poisson': ['APPLY_NOISE', 'BOOLEAN'],
    'binning': ['EMSOFT_BINNING_MODE', 'CONTROLLEDVOCABULARY'],
    'gammavalue': ['GAMMA_CORRECTION_FACTOR', 'REAL'],
    'maskpattern': ['APPLY_MASK', 'BOOLEAN'],
    'maskradius': ['MASK_RADIUS', 'REAL'],
    'hipassw': ['HI_PASS_FILTER', 'REAL'],
    'nregions': ['REGIONS_ADAPTIVE_HIST_EQUALISATION', 'INTEGER'],
    'scalingmode': ['EBSDSIM_INTENSITY_SCALING_MODE', 'CONTROLLEDVOCABULARY'],
}

master_pattern_metadata_mapping = {
    'BetheParameters': ['BETHE_PARAMETERS', 'VARCHAR'],
    'BetheParametersFile': ['BETHE_PARAM_FILENAME', 'VARCHAR'],
    'dmin': ['DMIN', 'REAL'],
    'npx': ['NPX', 'INTEGER'],
    'doLegendre': ['LAT_GRID_TYPE', 'CONTROLLEDVOCABULARY'],
    'energyfile': ['OUTPUT_FILENAME', 'VARCHAR'],
    'uniform': ['UNIFORM_MASTERPATTERN', 'BOOLEAN'],
    'useEnergyWeighting': ['ENERGY_WEIGHTING', 'BOOLEAN'],
}

monte_carlo_metadata_mapping = {
    'totnum_el': ['NUM_ELECTRONS', 'INTEGER'],
    'multiplier': ['ELECTRON_MULTIPLIER', 'INTEGER'],
    'Ebinsize': ['ENERGY_BIN_SIZE', 'REAL'],
    'Ehistmin': ['MIN_ENERGY', 'REAL'],
    'EkeV': ['HV', 'REAL'],
    'mode': ['EBSDSIM_MODE', 'CONTROLLEDVOCABULARY'],
    'dataname': ['OUTPUT_FILENAME', 'VARCHAR'],
    'depthmax': ['DEPTH_MAX', 'REAL'],
    'depthstep': ['DEPTH_STEP', 'REAL'],
    'xtalname': ['CRYSTAL_STRUCTURE_FILENAME', 'VARCHAR'],
    'numsx': ['NPIXELX', 'INTEGER'],
    'sig': ['TILTANGLE', 'REAL'],
    'omega': ['TILTANGLE_RD', 'REAL'],
    'sigstart': ['ANGLE_START', 'REAL'],
    'sigsend': ['ANGLE_END', 'REAL'],
    'sigsstep': ['ANGLE_STEP', 'REAL'],
    'ivolx': ['INTERACTION_VOLUME_X', 'INTEGER'],
    'ivoly': ['INTERACTION_VOLUME_Y', 'INTEGER'],
    'ivolz': ['INTERACTION_VOLUME_Z', 'INTEGER']
}


angle_mapping = {
    'sigstart' : ['ANGLE_START', 'REAL'],
    'sigend' : ['ANGLE_END', 'REAL'],
    'sigstep' : ['ANGLE_STEP', 'REAL'],
}


job_metadata_mapping = {
    'ProgramName': ['SOFTWARENAME', 'VARCHAR'],
    'UserName': ['User_Name', 'VARCHAR'],
    'Version': ['SOFTWAREVERSION', 'VARCHAR'],
    'StartTime': ['DATETIME', 'TIMESTAMP'],
    'StopTime': ['DATETIME_END', 'TIMESTAMP']
}


def add_metadata(metadata_dict: dict, mapping: dict, dataset: h5py.Dataset) -> None:
    """Parses dataset from h5 file to add metadata key-value pairs to dictionary"""
    for key, (openbis_code, openbis_datatype) in mapping.items():
        if key in dataset:
            value = dataset[key][0]
            if isinstance(value, bytes):
                value = value.decode()
            value = str(value)

            if openbis_datatype == "BOOLEAN":
                if value == 'n':
                    value = False
                elif value == 'y':
                    value = True
            if openbis_datatype == "REAL":
                value = float(value)
            if openbis_datatype == "INTEGER":
                value = int(float(value))
            if openbis_datatype == "CONTROLLEDVOCABULARY":
                value = map_to_vocabulary_term(key, value)
            if openbis_datatype == "TIMESTAMP":
                value = value.replace(' 0:', ' 12:')
                datetime_format = "%b %d %Y,  %I:%M:%S.%f %p"
                datetime_object = datetime.strptime(value, datetime_format)
                value = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
            metadata_dict[openbis_code] = value
            

def map_to_vocabulary_term(key:str, value:str) -> str:
    """Maps the values of vocabulary terms to openBIS codes"""
    if key == 'scalingmode':
        if value == 'not':
            value = 'NOSCALING'
        elif value == 'lin':
            value = 'LINEAR'
        elif value == 'gam':
            value = 'GAMMA'
    if key == 'doLegendre':
        if value == '.TRUE.' or value == '1':
            value = 'LEGENDRE'
        else:
            value = 'LAMBERT'
    if key == 'mode':
        if value == 'bse1':
            value = 'EMSOFT_MODE_BSE1'
        elif value == 'full':
            value = 'EMSOFT_MODE_FULL'
        elif value == 'Ivol':
            value = 'EMSOFT_MODE_IVOL'
    return value


def get_decimal_from_line(line: str):
    value = line.split(' = ')[-1]
    value = ''.join(c for c in value if c.isdigit() or c == '.')
    return value


def add_angle_metadata(metadata_dict: dict, mapping: dict, lines):
    for line in lines:
        for key, (openbis_code, openbis_datatype) in mapping.items():
            if key in line:
                metadata_dict[openbis_code] = get_decimal_from_line(line)        


def get_metadata(filename: str) -> dict:
    """Extracts metadata from a h5 file generated by EMsoft 

    Args:
        filename (str): (full) path to the h5 file from which the metadata are to be extracted

    Returns:
        dict: metadata dictionary corresponding to openBIS custom object types EBSD_SIM_INTERNAL, EBSD_SIM_MASTERPATTERN, EBSD_SIM_SCREENPATTERN"
    """


    md_dict = {}

    with h5py.File(filename, "r") as f:
    
        if 'EMEBSDNML' in list(f['NMLfiles']):
            # Metadata for screen patterns h5 file
            add_metadata(md_dict, job_metadata_mapping, f['EMheader']['EBSD']) 
            add_metadata(md_dict, screen_pattern_metadata_mapping, f['NMLparameters']['EBSDNameList'])
            # Add metadata from emdata
            md_dict['CRYSTAL_STRUCTURE_FILENAME'] = str(f['EMData']['EBSD']['xtalname'][0])[2:-1]
            md_dict['EULER_ANGLE_1'] = str(f['EMData']['EBSD']['EulerAngles'][1][0])
            md_dict['EULER_ANGLE_2'] = str(f['EMData']['EBSD']['EulerAngles'][1][1])
            md_dict['EULER_ANGLE_3'] = str(f['EMData']['EBSD']['EulerAngles'][1][2])

            md_dict['DATA_TYPE'] = md_dict['BIT_DEPTH']       
            md_dict['BIT_DEPTH'] = int("".join(c for c in md_dict['BIT_DEPTH']  if c.isdigit()))
            if not md_dict['APPLY_MASK']:
                md_dict['MASK_RADIUS'] = None

        elif 'EBSDmasterNML' in list(f['NMLfiles']):
            # Metadata for master patterns h5 file      
            add_metadata(md_dict, job_metadata_mapping, f['EMheader']['EBSDmaster'])
            add_metadata(md_dict, monte_carlo_metadata_mapping, f['NMLparameters']['MCCLNameList'])
            add_metadata(md_dict, master_pattern_metadata_mapping, f['NMLparameters']['EBSDMasterNameList'])

            nml_files = [str(line)[3:-1] for line in list(f['NMLfiles']['MCOpenCLNML'])]
            add_angle_metadata(md_dict, angle_mapping, nml_files)

            md_dict['BETHE_PARAMETERS'] = ', '.join(map(str,f['EMData']['EBSDmaster']['BetheParameters']))

        else:
            # Metadata for simulation patterns h5 file
            add_metadata(md_dict, job_metadata_mapping, f['EMheader']['MCOpenCL'])
            add_metadata(md_dict, monte_carlo_metadata_mapping, f['NMLparameters']['MCCLNameList'])

            nml_files = [str(line)[3:-1] for line in list(f['NMLfiles']['MCOpenCLNML'])]
            add_angle_metadata(md_dict, angle_mapping, nml_files)

    return md_dict
