# These are the keys of the dictionary that are used in the definiton of the datatype in openBIS
# I.e. the automatic metadata extraction needs to fill these keys and pass them on to openBIS when registering a file
# of type "SE" (=SEM image)
keys_SEMDict = ['Company', 'Device', 'DeviceModel', 'ImageStripSize', 
        'Magnification', 'MagnificationReference',
        'PixelSizeX', 'PixelSizeY', 'SoftwareVersion',
        'Date', 'Time', 'ViewFieldsCountX', 'ViewFieldsCountY',
        'BeamCurrent',
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
