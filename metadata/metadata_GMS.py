from pathlib import Path
import rsciio.digitalmicrograph as dm
from datetime import datetime
from .helpers import *
from decimal import Decimal, getcontext
import numpy as np # For preview


def get_image_stack(path, vertical=True):
    path = Path(path)
    ext = path.suffix[1:]
    if ext in ['dm3', 'dm4']:
        file_reader = dm.file_reader
    else:
        raise TypeError(f'This parser is intended for dm3 / dm4 files!')
    try:
        file_content = file_reader(path)
    except Exception as e:
        raise TypeError(f'RosettaSciIO was not able to parse the {ext} file!')
    arrays = []
    for dataset in file_content:
        if 'data' in dataset:
            img = dataset['data']
            if img.ndim == 3:
                img = img[-1]
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255
            img = img.astype('uint8')
            arrays.append(img)
    if len(arrays):
        if vertical:
            arrays = np.vstack(arrays)
        else:
            arrays = np.hstack(arrays)
    else:
        raise RuntimeError(f'Empty file: {path.name}')
    return np.vstack(arrays)    


def get_metadata(path):
    path = Path(path)
    ext = path.suffix[1:]
    if ext in ['dm3', 'dm4']:
        file_reader = dm.file_reader
    else:
        raise TypeError(f'This parser is intended for dm3 / dm4 files!')
    try:
        file_content = file_reader(path)
    except Exception as e:
        raise TypeError(f'RosettaSciIO was not able to parse the {ext} file!')

    num_datasets = len(file_content)
    
    if num_datasets > 1:
        raw_metadata = {}
        for dataset_idx, dataset in enumerate(file_content):
            if 'original_metadata' in dataset and 'metadata' in dataset:
                raw_metadata.update({
                    f'Dataset{dataset_idx}.' + k: v
                        for k, v in dataset['original_metadata'].items()
                })
                raw_metadata.update({
                    f'Dataset{dataset_idx}.' + k: v
                        for k, v in dataset['metadata'].items()
                })
    elif num_datasets == 1:
        dataset = file_content[0]
        raw_metadata = {
            **dataset['metadata'],
            **dataset['original_metadata']
        }

    dataset = file_content[0]
    if 'data' in dataset:
        ndim = dataset['data'].ndim
        if ndim == 2:
            num_frames = 1
            num_channels = 1
        elif ndim == 3:
            num_frames = dataset['data'].shape[0]
            num_channels = 1
        else:
            raise ValueError('4D image')
        data_type = str(dataset['data'].dtype)
        bitdepth_str = data_type.replace('uint', '').replace('int', '')
        bitdepth_str = bitdepth_str.replace('float', '')
        try:
            bitdepth = int(bitdepth_str)
        except ValueError:
            raise ValueError('Cannot convert Bit Depth to int !')

    del dataset, file_content
    metadata_dict = clean(
        raw_metadata, remove_prefix='ImageList.TagGroup0.Image'
    )

    metadata_dict['NumberOfDatasets'] = num_datasets
    metadata_dict['NumberOfChannels'] = num_channels
    metadata_dict['NumberOfFrames'] = num_frames
    metadata_dict['DataType'] = data_type
    metadata_dict['BitDepth'] = bitdepth

    return metadata_dict


def filter_metadata(md_dict: dict, keys: list):
    field_map = {
        'Data.Dimensions.Data0':
        'DATA_SIZE_X',
        'Data.Dimensions.Data1':
        'DATA_SIZE_Y',
        'Tags.MicroscopeInfo.ImagingMode':
        'TEM_IMAGING_MODE',
        'Tags.Acquisition.Device.Source':
        'DETECTOR1_NAME',
        'Tags.Acquisition.Device.Temperature[C]':
        'DETECTOR1_TEMP',
        # 'Tags.Acquisition.Parameters.Detector.Exposure[s]': # bad for dm3
        'Tags.DataBar.ExposureTime[s]':
        'EXPOSURE_TIME',
        'Tags.MicroscopeInfo.STEMCameraLength':
        'CAMERA_LENGTH_MM',
        'Tags.MicroscopeInfo.IndicatedMagnification':
        'MAGNIFICATION_REAL',
        'Tags.MicroscopeInfo.ActualMagnification':
        'MAGNIFICATION_ACTUAL',
        'Tags.MicroscopeInfo.StagePosition.StageX':
        'STAGE_Y_UM',
        'Tags.MicroscopeInfo.StagePosition.StageY':
        'STAGE_Y_UM',
        'Tags.MicroscopeInfo.StagePosition.StageZ':
        'STAGE_Z_UM',
        'Tags.MicroscopeInfo.StagePosition.StageAlpha':
        'STAGE_TILT_A',
        'Tags.MicroscopeInfo.StagePosition.StageBeta':
        'STAGE_TILT_B',
        'Tags.MicroscopeInfo.ProbeCurrent[nA]': 'BEAM_CURRENT_NA',
        'Tags.MicroscopeInfo.EmissionCurrent[µA]': 'EMISSION_CURRENT_UA',
        'Tags.MicroscopeInfo.Specimen': 'SPECIMEN_NAME',
        'Tags.GMSVersion.Created': 'SOFTWAREVERSION', 
        'Tags.MicroscopeInfo.ProbeSize[nm]': 'SPOT_SIZE',
        'Tags.MicroscopeInfo.Name': 'DEVICE_MANUFACTURER',
        'Tags.MicroscopeInfo.Voltage': 'ACCELERATING_VOLTAGE',
        'Tags.Acquisition.Frame.Sequence.FrameTime[ns]': 'FRAME_TIME',
        'Tags.MicroscopeInfo.IlluminationMode': 'IS_STEM',
        'BitDepth': 'BIT_DEPTH',
        'DataType': 'DATA_TYPE',
        'NumberOfFrames': 'NUMBER_OF_FRAMES',
        'NumberOfChannels': 'NUMBER_OF_CHANNELS',
        'NumberOfDatasets': 'NUMBER_OF_DATASETS',
    }
    out_dict = dict.fromkeys(keys)

    # Set precision
    getcontext().prec = 10

    # Datetime
    date_str = md_dict.get('Tags.DataBar.AcquisitionDate')
    time_str = md_dict.get('Tags.DataBar.AcquisitionTime')
    if date_str and time_str:
        date_obj = datetime.strptime(date_str, '%m/%d/%Y').date()
        if 'AM' in time_str or 'PM' in time_str:
            time_obj = datetime.strptime(time_str, '%I:%M:%S %p').time()
        else:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        datetime_obj = datetime.combine(date_obj, time_obj)
        out_dict['DATETIME'] = datetime_obj.isoformat()
        out_dict['DATE'] = date_obj.isoformat()
        out_dict['TIME'] = time_obj.isoformat()

    # Alternative keys for Microscope Name
    for old_key in ['Tags.SessionInfo.Microscope',
                    'Tags.MicroscopeInfo.Microscope']:
        device_name = md_dict.get(old_key)
        if device_name is not None and len(device_name):
            out_dict['DEVICE_NAME'] = device_name
            break
    
    pixel_x = md_dict.get('Data.Calibrations.Dimension.TagGroup0.Scale')
    pixel_y = md_dict.get('Data.Calibrations.Dimension.TagGroup1.Scale')
    unit_x = md_dict.get('Data.Calibrations.Dimension.TagGroup0.Units')
    unit_y = md_dict.get('Data.Calibrations.Dimension.TagGroup1.Units')
    
    if unit_x == unit_y:
        if unit_x.startswith('1/'):
            out_dict['PIXEL_SIZE_X'] = convert_value(pixel_x, 'nm', unit_x[2:])
            out_dict['PIXEL_SIZE_Y'] = convert_value(pixel_y, 'nm', unit_x[2:])
            out_dict['PIXEL_UNIT_X'] = out_dict['PIXEL_UNIT_Y'] = '1/nm'
        else:
            out_dict['PIXEL_SIZE_X'] = convert_value(pixel_x, unit_x, 'nm')
            out_dict['PIXEL_SIZE_Y'] = convert_value(pixel_y, unit_x, 'nm')
            out_dict['PIXEL_UNIT_X'] = out_dict['PIXEL_UNIT_Y'] = 'nm'
    else:
        raise ValueError('Different units used for pixels')

    # Copy remaining
    for old_key in field_map:
        out_dict[field_map[old_key]] = md_dict.get(old_key)

    # ####################### Convert Numerical Values #######################
    
    # Convert High Tension/Voltage from volts to kilovolts
    hv = out_dict.get('ACCELERATING_VOLTAGE')
    if hv: out_dict['ACCELERATING_VOLTAGE'] = convert_value(hv, '1V', 'kV')

    # Convert frame time from nanoseconds to seconds
    frame_time = out_dict.get('FRAME_TIME')
    if frame_time:
        out_dict['FRAME_TIME'] = convert_value(frame_time, 'ns', '1s')

    # ############################### Clean up ###############################

    # Flag for STEM images 
    is_stem = out_dict.get('IS_STEM')
    if is_stem == 'TEM':
        is_stem = False
    else:
        is_stem = None
    out_dict['IS_STEM'] = is_stem

    # Control for Microscope Manufacturer
    manufacturer = out_dict.get('DEVICE_MANUFACTURER')
    if manufacturer:
        for name in ['JEOL', 'ZEISS', 'FEI', 'HITATCHI']:
            if name.lower() in manufacturer.lower():
                manufacturer = name
                out_dict['DEVICE_MANUFACTURER'] = manufacturer
                break
    
    # Collapse MAG1 and MAG2 to Imaging
    imaging_mode = out_dict.get('TEM_IMAGING_MODE')
    if imaging_mode:
        if 'diff' in imaging_mode.lower():
            imaging_mode = 'Diffraction'
        if 'mag' in imaging_mode.lower():
            imaging_mode = 'Imaging'
    out_dict['TEM_IMAGING_MODE'] = imaging_mode

    # Camera is an imaging detector
    out_dict['DETECTOR1_TYPE'] = 'ImagingDetector'

    # Add Software Name Number to Software Version
    software_version = out_dict.get('SOFTWAREVERSION')
    if software_version:
        software_version = 'Gatan Microscopy Suite ' + software_version 
        out_dict['SOFTWAREVERSION'] = software_version 
    
    # Remove manufacturer name from device name
    device_name = out_dict.get('DEVICE_NAME')
    if device_name and manufacturer:
        device_name = device_name.replace(manufacturer, '').strip()
        out_dict['DEVICE_NAME'] = device_name

    # Clean specimen name
    specimen_name = out_dict.get('SPECIMEN_NAME')
    if isinstance(specimen_name, list):
        if len(specimen_name) == 0:
            specimen_name = None
        else:
            specimen_name = ','.join(specimen_name)
    out_dict['SPECIMEN_NAME'] = specimen_name

    # ############################## Sanity Check #############################
    set_a = set(out_dict.keys())
    set_b = set(keys)
    assert set_a == set_b, print('Problem:', set_a.symmetric_difference(set_b))

    return out_dict

def parse(filename: str, keys: list) -> dict:
    d = get_metadata(filename)
    d = filter_metadata(d, keys)
    return d 
