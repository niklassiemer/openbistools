from datetime import date, datetime

from keys_SEMDict import keys_SEMDict

def get_metadata(file):
    """Extract metadata from a *.tif file.
       The metadata are at the beginning of the file, one key-value pair per line.
       We open the file in binary mode and extract all relevant keys.
       This implies that all comparisons have to be made using binary format as well, and the values 
       we found have to be converted/decoded from binary.

    :param file: Filename of the *.tif file 
    :return: dictionary of extracted metadata
    :rtype: dictionary
    """
    #list all the keys that we need to extract from the TIF
    # only one pixel size (AP_PIXEL_SIZE)?  but different from AP_IMAGE_PIXEL_SIZE?
    # note: All key-value pairs separated by "=", except for Time and Date (separated by ":") - however,
    #       the Software version has a ":" as part of the value....
    #       some values have units that need to be stripped.
    keys_ZeissLeo = ['Version', 'Time', 'Date',
                    'EHT = ', 'WD =', 'Beam Current',
                    'Gun Shift X', 'Gun Shift Y', 'Gun Tilt X', 'Gun Tilt Y',
                    'Pixel Size', 'Mag =', 'Reference Mag',
                    'System Vacuum', 'ScanGen Type', 'Line Time',
                    'Focus = ',
                    'Detector',
                    'Stigmation X', 'Stigmation Y',
                    'Stage at X', 'Stage at Y', 'Stage at Z', 'Stage at T', 'Stage at R'
                    ]

    #N.B. vacuum is in mbar, may need to convert to Pa?

    metadata_tif = dict.fromkeys(keys_ZeissLeo)
    
    # this is the return object where we pass the metadata to openBIS.
    # we may need to translate some terms from the ones we extract from the TIF
    metadata = dict.fromkeys(keys_SEMDict)

    with open(file, 'rb') as f:
        for num, line in enumerate(f.readlines(), 1):
            #problematic unicode signs to remove:
            # \xb0 (degree), 0xb5 (mu)
            line = line.replace(b'\xb0',b'')
            line = line.replace(b'\xb5',b'')
            for key in keys_ZeissLeo:
                if (line.startswith(key.encode('ascii'))):
                    # Date and Time are separated by ":", all others by "="
                    if key=='Date' or key=='Time':
                        delim=b':'
                    else :
                        delim=b'='
                    value = (line.split(delim,1)[1])
                    value = value.decode('ascii').strip()
                    # strip units
                    if key != 'Time' and key != 'Date' and key != 'Version' :
                        value = value.split(' ',1)[0]
                    metadata_tif[key] = value


    # convert Time/Date into Datetime
    datetime_str = metadata_tif['Date'] + ' ' + metadata_tif['Time']
    datetime_object = datetime.strptime(datetime_str, '%d %b %Y %H:%M:%S')
    metadata_tif['DateTime'] = datetime_object

    #
    # now copy over metadata information with the corresponding keys for the openBIS data type
    #

    #static information not in the TIF
    metadata['Company']                 = 'Zeiss'
    metadata['Device']                  = 'LEO'
    metadata['DeviceModel']             = '1530'

    metadata['SoftwareVersion']         = metadata_tif['Version']
    metadata['ScanMode']                = metadata_tif['ScanGen Type']
    metadata['ScanSpeed']               = metadata_tif['Line Time']

    metadata['Magnification']           = metadata_tif['Mag =']
    metadata['MagnificationReference']  = metadata_tif['Reference Mag']

    metadata['PixelSizeX']              = metadata_tif['Pixel Size']
    metadata['PixelSizeY']              = metadata_tif['Pixel Size']

    metadata['Date']                    = datetime_object.date()
    metadata['Time']                    = datetime_object.time()
    metadata['DateTime']                = metadata_tif['DateTime']

    metadata['ChamberPressure']         = metadata_tif['System Vacuum']
   
    metadata['HV']                      = metadata_tif['EHT = ']
    metadata['BeamCurrent']             = metadata_tif['Beam Current']

    metadata['GunShiftX']               = metadata_tif['Gun Shift X']
    metadata['GunShiftY']               = metadata_tif['Gun Shift Y']
    metadata['GunTiltX']                = metadata_tif['Gun Tilt X']
    metadata['GunTiltY']                = metadata_tif['Gun Tilt Y']

    metadata['StageX']                  = metadata_tif['Stage at X']
    metadata['StageY']                  = metadata_tif['Stage at Y']
    metadata['StageZ']                  = metadata_tif['Stage at Z']
    metadata['StageTilt']               = metadata_tif['Stage at T']
    metadata['StageRotation']           = metadata_tif['Stage at R']

    metadata['Detector0']               = metadata_tif['Detector']
    metadata['WD']                      = metadata_tif['WD =']
    metadata['DepthOfFocus']            = metadata_tif['Focus = ']

    metadata['StigmatorX']              = metadata_tif['Stigmation X']
    metadata['StigmatorY']              = metadata_tif['Stigmation Y']

    return metadata