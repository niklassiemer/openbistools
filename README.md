# Register linked data
OpenBis has two ways of keeping track of files and their metadata: Either they are stored internally by the Data Store Server (DSS). Here, OpenBIS ingests the files and handles their storage and delivery to the user itself.
Alternatively, data can be stored on an external data storage server. This is exploited using the [Big Data Link Services](https://openbis.readthedocs.io/en/latest/app-openbis-command-line/README.html#big-data-link-services). This tool is based on ```git``` and [git-annex](https://git-annex.branchable.com/).

Depending on the application scenario, this may or may not fit all needs.

The code here is built on the same API calls and PyBIS code but does not make an assumption on where the data are as such - although for our specific use-case, we upload the data to an S3-based storage solution. Therefore, related convenience functions are provided and also called from the main script, assuming that the user will want to upload the data to an S3 storage.

The general configuration (OpenBIS server details, S3 details) can be stored in a config file (compatible with the standard ConfigParser), additional arguments can be given via the command line.

The metadata extraction scripts are specific to our instruments, you will need to remove this or modify the scripts, depending on your setup.



| Content                        | Description |
|--------------------------------|-------------|
| ```Hello.py``` and ```pages``` | Streamlit multi page app for up-/downloading files from/to S3 and linking to openBIS |
| ```registerLinkData.py```      | Python script to upload files to S3 and link them to openBIS |
| ```s3_tools.py```              | collection of tools to interact with S3 / Coscine |
| ```pybis_tools.py```           | PyBIS code needed for linking data to openBIS |
| ```metadata```                 | extraction scripts for metadata information from various file formats |
| ```keys```                     | metadata schema definitions (defined [here](https://git.rwth-aachen.de/Kerzel/openbis_materialsscience_schema))|


## Data Upload Tool
To run the data upload web tool, create a virtual environment (`Python 3.10`) and activate it.

The files are temporarily stored in `/path/to/temporary/directory`.

`MAX_FILE_SIZE_MB` controls the maximum allowed file size (and the total size of files that can be downloaded).

```
python -m venv .venv
source .venv/bin/activate
python -m pip install poetry
python -m poetry install
cd rdm
python -m streamlit run Hello.py --server.port PORT --server.maxUploadSize MAX_FILE_SIZE_MB  -- --temp_dir "/path/to/temporary/directory" # Not ./Hello.py
```

## Uploading data using script
To use the script to upload files, create a virtual environment (`Python 3.10`) and activate it.

```
python -m venv .venv
source .venv/bin/activate
python -m pip install poetry
python -m poetry install
cd rdm

export OBIS_TOKEN="SZNqHsCcLGwYVpFXkqJHpmoboltjgzQjrONTRmjtlSNIHXqIAl"

python registerLinkData.py --mode s3 --config_file /path/to/config/file -e ENTRY_ID -f /path/to/file -t DATASET_TYPE -parser PARSER_CODE -prefix OPTIONAL_PREFIX -n DATASET_NAME
```