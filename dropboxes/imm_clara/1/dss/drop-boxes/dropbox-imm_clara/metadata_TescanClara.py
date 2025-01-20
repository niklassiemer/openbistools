from datetime import date, datetime

keys_TescanClara = ['Company', 'Device', 'DeviceModel', 'ImageStripSize', 
        'Magnification', 'MagnificationReference',
        'PixelSizeX', 'PixelSizeY', 'SoftwareVersion',
        'Date', 'Time', 'ViewFieldsCountX', 'ViewFieldsCountY',
        'AcceleratorVoltage', 'BeamCurrent',
        'ChamberPressure', 'DepthOfFocus', 
        'Detector0', 'Detector0Gain' , 'Detector0Offset',
        'DwellTime', 'EmissionCurrent',
        'GunShiftX', 'GunShiftY', 'GunTiltX', 'GunTiltY',
        'HV', 'PotentialMode', 'PrimaryDetectorGain',
        'PrimaryDetectorOffset', 'ScanMode',
        'ScanRotation', 'ScanSpeed',
        'SessionID', 'SpotSize',
        'StageRotation', 'StageTilt',
        'StageX', 'StageY', 'StageZ',
        'StigmatorX', 'StigmatorY',
        'VirtualObserverDistance', 'WD'
]

def get_metadata(file, keys = keys_TescanClara):
    """Extract metadata from a *.tif file recorded by the Tescan Clara microscope.
       The information is contained both in the *.tif file, as well a separate ASCII *.hdr file
       However, it might not be possible to work on both files or have both files available, so we
       only use the *.tif file. The relevant metadata is right at the end of the file, where each 
       metadata parameter is in one line in the format key=value.
       Unfortunately, Tescan does not seem to use TIFF image tags for these, so we have to parse
       the file as binary. (mode 'rb' - read only, binary).
       This implies that all comparisons have to be made using binary format as well, and the values 
       we found have to be converted/decoded from binary.

    :param file: Filename of the *.tif file recorded by a Tescan Clara
    :type file: str
    :param keys: metadata keys that should be extracted from the image , defaults to keys_TescanClara
    :type keys: list of strings, optional
    :return: dictionary of extracted metadata
    :rtype: dictionary
    """

    metadata = {}

    with open(file, 'rb') as f:
        for num, line in enumerate(f.readlines(), 1):
            for key in keys:
                if (line.startswith(key.encode('ascii'))):
                    value = (line.split(b'=',1)[1])
                    value = value.decode('ascii').strip()
                    metadata[key] = value

    # convert Time/Date into Datetime
    datetime_str = metadata['Date']+' '+metadata['Time']
    datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    metadata['DateTime'] = datetime_object
    
    return metadata