from datetime import date, datetime

# in Python 3, ConfigParser was renamed to configparser
#import configparser
from ConfigParser import ConfigParser

# handle strings as files
import StringIO


from keys_SEMDict import keys_SEMDict


def get_metadata(file,keys_SEMDict = keys_SEMDict):
    """Extract metadata from a *.tif file recorded by the FEI Helios  microscope.
       The FEI Helios (at least in the present version) includes the metadata in the TIF file akin to the .ini format
       recognised by the python configparser, i.e.
       [Field1]
       property1 = value1
       property2 = value2

       [Field2]
       property1 = value1

       etc.

       The metadata block is at the end of the file and starts with the tag '[User]'. In a first step, we split the TIF file
       at this tag (and prepend it again), so that we are left with the configuration-like metadata.
    

    :param file: Filename of the *.tif file 
    :type file: str
    :param keys: metadata keys for the dictionary to be filled with all entries in the openBIS data type
    :type keys: list of strings, optional
    :return: dictionary of extracted metadata
    :rtype: dictionary
    """

    metadata = dict.fromkeys(keys_SEMDict)


    with open(file, 'rb') as f:
        raw_metadata = f.read().split(b'[User]')[1].decode('ascii')
    raw_metadata = '[User]\r\n' + raw_metadata
    # remove \x00 end-of-string
    raw_metadata = raw_metadata.rstrip('\x00')
     
    config = ConfigParser()

    # "read_string" does not exist yet in the Jython / Python 2.7 version....
    # config.read_string(raw_metadata)

    # convert string and pretend it's a file
    buf = StringIO.StringIO(raw_metadata)
    config.readfp(buf)


    #sections = config.sections()

    metadata['Company']             = 'FEI'
    # there appears to be a spurious " in the text...?
    metadata['Device']              = config.get('System','SystemType').replace('"','')
    metadata['DeviceModel']         = config.get('System','Type')
    metadata['SoftwareVersion']     = config.get('System','Software')

    metadata['Date']                = config.get('User','Date')
    metadata['Time']                = config.get('User','Time')
    # convert Time/Date into Datetime
    datetime_str                    = metadata['Date'] + ' ' + metadata['Time']
    datetime_object                 = datetime.strptime(datetime_str, '%m/%d/%Y %I:%M:%S %p')
    metadata['DateTime']            = datetime_object

    #metadata['AcceleratorVoltage']  = config.get('Beam','HV')
    metadata['HV']                  = config.get('Beam','HV')
    metadata['StigmatorX']          = config.get('Beam','StigmatorX')
    metadata['StigmatorY']          = config.get('Beam','StigmatorY')
    metadata['ScanRotation']        = config.get('Beam','ScanRotation')
    metadata['GunShiftX']           = config.get('Beam','BeamShiftX')
    metadata['GunShiftY']           = config.get('Beam','BeamShiftY')
    try:
        metadata['BeamCurrent']         = config.get('EBeam','BeamCurrent')
    except KeyError as e:
        pass

    metadata['ChamberPressure']     = config.get('Vacuum','ChPressure')

    metadata['StageX']              = config.get('Stage','StageX')
    metadata['StageY']              = config.get('Stage','StageY')
    metadata['StageZ']              = config.get('Stage','StageZ')
    metadata['StageTilt']           = config.get('Stage','StageT')
    metadata['StageRotation']       = config.get('Stage','StageR')
    metadata['WD']                  = config.get('Stage','WorkingDistance')

    metadata['PixelSizeX']          = config.get('Scan','PixelWidth')
    metadata['PixelSizeY']          = config.get('Scan','PixelHeight')

    metadata['Detector0']           = config.get('Detectors','Name')

    return metadata
