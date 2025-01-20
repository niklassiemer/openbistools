#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Copyright [2024] [Ulrich Kerzel, Khalil Rejiba]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

    Collection of tools and helper functions that are used in handling data,
    extracting metadata, etc.

"""

import logging

# Scripts to extract metadata from files to be linked
from metadata.metadata_TescanClara import get_metadata as parse_tescan_clara
from metadata.metadata_FEIHelios import get_metadata as parse_fei_helios
from metadata.metadata_ZeissLeo import get_metadata as parse_zeiss_leo
from metadata.metadata_ZeissGemini import get_metadata as parse_zeiss_gemini
from metadata.metadata_ZeissSupra import get_metadata as parse_zeiss_supra
from metadata.metadata_EdaxTeam_zip import get_metadata as parse_edaxteam_zip
from metadata.metadata_EMsoft_h5 import get_metadata as parse_emsoft_h5
from metadata.metadata_Velox import parse as parse_emd_velox
from metadata.metadata_GMS import parse as parse_gms_dm

# Keys defines as OprnBIS properties
from metadata.keys.keys_SEMDict import keys_SEMDict
from metadata.keys.keys_TEMDict import keys_TEMDict



def get_metadata(file_name_fqdn : str,
                 data_set_type : str,
                 metadata_parser : str,
                 convert_lowercase : bool = True) -> dict:
    """ Use the specified parser to extract metadata from the file.

    The fields of the metadata extracted here have to correspond 
    to the variables specified in openBIS, both name and variable type.
    OpenBIS requires the metadata dictionary to be lowercase.

    Args:
        file_name_fqdn (str): name and path to the file
        data_set_type (str): openBIS data set type.
        metadata_parser (str): identifier of the parser to use
        convert_lowercase (bool): flag to convert metadata keys to lowercase

    Returns:
        dict: dictionary holding the metadata exteracted from the file
    """

    logging.debug('Get metadata: File: {}, DatasetType: {}, Parser: {}'.format(
        file_name_fqdn, data_set_type, metadata_parser
    ))

    # Initialize the metadata dictionary
    metadata_dict = dict()
    try:
        if data_set_type == 'SEM_DATA':
            keys = keys_SEMDict
            match metadata_parser:
                case 'IMM_Clara' | 'IMM Tescan Clara':
                    metadata_dict = parse_tescan_clara(file_name_fqdn, keys)
                case 'IMM_FEIHelios' | 'IMM FEI Helios' | 'MCh FEI Helios' | 'MPIE FEI Helios' | 'MPIE FEI Scios':
                    metadata_dict = parse_fei_helios(file_name_fqdn, keys)
                case 'IMM_ZeissLeo' | 'IMM Zeiss Leo':
                    metadata_dict = parse_zeiss_leo(file_name_fqdn, keys)
                case 'GFE_ZeissGemini' | 'GfE Zeiss Gemini 300':
                    metadata_dict = parse_zeiss_gemini(file_name_fqdn, keys)
                case 'KKS Zeiss Supra 55 VP':
                    metadata_dict = parse_zeiss_supra(file_name_fqdn, keys)
        elif data_set_type == 'TEM_DATA':
            keys = keys_TEMDict
            match metadata_parser:
                case 'Thermo Scientific Velox Software emd' | 'VELOX_EMD':
                    metadata_dict = parse_emd_velox(file_name_fqdn, keys)
                case 'Gatan Microscopy Suite dm3/dm4' | 'GMS_DM':
                    metadata_dict = parse_gms_dm(file_name_fqdn, keys)
        elif data_set_type == 'EBSD_EXP_DATA':
            match metadata_parser:
                case 'EDAX_ZIP' | 'EDAX APEX zip':
                    metadata_dict = parse_edaxteam_zip(file_name_fqdn)
        elif data_set_type == 'EBSD-EDS_DATA':
            match metadata_parser:
                case 'EDAX_ZIP' | 'EDAX APEX zip':
                    metadata_dict = parse_edaxteam_zip(file_name_fqdn)
        elif data_set_type == 'EBSD_SIM_INTERNAL':
            match metadata_parser:
                case 'EMSOFT_H5':
                    metadata_dict = parse_emsoft_h5(file_name_fqdn)
        elif data_set_type == 'EBSD_SIM_MASTERPATTERN':
            match metadata_parser:
                case 'EMSOFT_H5':
                    metadata_dict = parse_emsoft_h5(file_name_fqdn)
        elif data_set_type == 'EBSD_SIM_SCREENPATTERN':
            match metadata_parser:
                case 'EMSOFT_H5':
                    metadata_dict = parse_emsoft_h5(file_name_fqdn)

        logging.debug('Metadata extraced from file: {}'.format(metadata_dict)) 

        if convert_lowercase:
            # Convert to lowercase
            old_keys = list(metadata_dict.keys())
            for key in old_keys:
                new_key = key.lower() 
                metadata_dict[new_key] = metadata_dict.pop(key)       
        
            logging.debug('Metadata (lowercase): {}'.format(metadata_dict))
    except Exception as e:
    
        logging.debug('Error while parsing metadata\nFile: {}, DatasetType: {}, Parser: {}\nError: {e}'.format(
            file_name_fqdn, data_set_type, metadata_parser
        ))

    return metadata_dict
