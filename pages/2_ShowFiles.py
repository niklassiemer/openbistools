import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
from zipfile import ZIP_DEFLATED, ZipFile
import requests
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import warnings


from s3_tools import s3_file_exists, s3_file_upload
from pybis_tools import get_info_from_identifier

# Utility to update experiment list
from Hello import find_relevant_locations

# Utilities for preview
from visualization.tem import get_image_stack as get_tem_stack
from visualization.video_rendering import VideoRenderer


warnings.filterwarnings(action="ignore", category=FutureWarning)

## ============================================================================
##  Helper Functions
## ============================================================================


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor

    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(required=True),
            "Preview": st.column_config.ImageColumn(width="small"),
            "S3 Path": st.column_config.TextColumn(width="small"),
            "Registration Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "Kind": None,
            "Size (Mb)": st.column_config.NumberColumn()
        },
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column

    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop("Select", axis=1)


def generate_preview(pil_img: Image, img_height: int = 480, format: str = "JPEG"):
    buffer = BytesIO()
    if format != "GIF":
        w, h = pil_img.size
        img = np.array(pil_img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-20)
        img = (img * 255).astype("uint8")
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(
            size=(int(img_height*w/h), img_height),
            resample=Image.Resampling.BOX,
        )
        pil_img = pil_img.convert('RGB')
        pil_img.save(buffer, format=format)
    else:
        pil_img.save(buffer, format=format, save_all=True)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    preview = f"data:image/{format.lower()};base64,{img_base64}"
    pil_img.close()
    return preview


## ============================================================================
##  Main app
## ============================================================================


st.set_page_config(
    page_title="Find data in openBIS",
    page_icon="media/SFB1394_icon.jpg",
    layout="wide",
)

try:
    _ = st.session_state.setup_done
    if "df_files" not in st.session_state:
        st.session_state.df_files = pd.DataFrame()
    if "df_all" not in st.session_state:
        st.session_state.df_all = pd.DataFrame()
    if "selection" not in st.session_state:
        st.session_state.selection = pd.DataFrame()
    if "selection_deletion" not in st.session_state:
        st.session_state.selection_deletion = pd.DataFrame()
    if "requested_deletion" not in st.session_state:
        st.session_state.requested_deletion = False
except AttributeError as e:
    st.switch_page("Hello.py")
st.sidebar.image("media/SFB1394_TitleImage_Cropped.png")
st.sidebar.write("Logged into openBIS: ", st.session_state.logged_in)
st.sidebar.write("OpenBIS Upload OK: ", st.session_state.openbis_upload_allowed)
st.sidebar.write("S3 Upload OK: ", st.session_state.s3_upload_ok)
st.sidebar.write("S3 Download OK: ", st.session_state.s3_download_ok)
st.sidebar.write("S3 Resource: ", st.session_state.obis_dmscode)


temp_dir = st.session_state.temp_dir
files_to_download = []
preview_dict = {
    "CRYS-STRUCT_DATA": "structure",
    "NANOINDENTATION_DATA": "nanoindentation",
    "PILLAR_COMPRESSION_DATA": "micropillarcompression",
}


st.title("Show Linked Data in openBIS")
st.write(
    """
    - If you select multiple files to download, they will be compressed into a zip archive.  
    - A CSV file containing the metadata is included in the zip archive.  
    - Activating the preview feature can significantly delay the rendering of the results.  
    - Metadata can also be fetched.
    """
)

with st.form("Experiment"):

    refresh_btn = st.form_submit_button(
        "Refresh",
        type="primary",
        disabled=not st.session_state.setup_done,
        help="Click if you added ELN entries after login",
    )

    if refresh_btn:
        with st.spinner("Please wait ..."):
            allowed_object_types = find_relevant_locations(
                username=st.session_state.openbis_username,
                include_samples=st.session_state.include_samples,
            )
        allowed_object_types_display = [
            " ".join(part.capitalize() for part in code.split("_"))
            for code in allowed_object_types
        ]
        allowed_object_types_display_str = ", ".join(allowed_object_types_display)
        st.warning(
            "You can upload to a Default Experiment "
            f"or one of following: {allowed_object_types_display_str}"
        )
    # Prompt users to choose an object or an experiment/collection

    st.selectbox(
        "Find the entry where your data is linked",
        st.session_state.experiment_name_list,
        index=None,
        placeholder="Select ELN entry",
        key="experiment",
        disabled=not st.session_state.logged_in,
    )
    include_preview = st.toggle(
        "Preview",
        help="Only if you're not in a hurry",
        disabled=not st.session_state.setup_done,
    )
    choose_exp_btn = st.form_submit_button(
        "Confirm",
        type="primary",
        disabled=not st.session_state.setup_done,
    )


    if choose_exp_btn and st.session_state.experiment is not None:

        st.session_state.selection_deletion = pd.DataFrame()

        with st.spinner(text="Preparing files ..."):
            identifier = st.session_state.experiments[st.session_state.experiment]
            exp_name, object_name = get_info_from_identifier(identifier)

            datasets = st.session_state.oBis.get_datasets(
                experiment=exp_name,
                sample=object_name,
            )

            format_func = lambda code: st.session_state.oBis.get_property_type(code).label
            
            md_keys = set()

            for ds in datasets:
                md_keys.update(ds.p.all().keys())
            if "$name" in md_keys:
                md_keys.remove("$name")  # Already in dataframe
            md_keys = sorted(md_keys, key= format_func)

            options = st.multiselect(
                label="Choose metadata fields to display "
                "(re-click on Confirm to update table)",
                options=md_keys,
                key="metadata_options",
                format_func=lambda code: st.session_state.oBis.get_property_type(
                    code
                ).label,
            )

            cols = [
                "Preview",
                "Name",
                "Type",
                "Kind",
                "Registration Date",
                "permID",
                "Filename",
                "Extension",
                "S3 Path",
                "Size (Mb)",
            ]

            cols_all = cols + [st.session_state.oBis.get_property_type(key).label for key in md_keys]

            for opt in options:
                opt = st.session_state.oBis.get_property_type(opt).label
                cols.append(opt)

            if datasets is not None and include_preview:
                n_files = len(datasets.df)
                if n_files > 500:
                    st.warning(f"Skipping preview, too many files ({n_files})")
                    include_preview = False
            
            data = []
            data_all = []
            for ds in datasets:
                name = ds.p["$name"]
                permID = ds.permId
                attrs_dict = ds.attrs.all()
                dataset_type = attrs_dict["type"]
                kind = attrs_dict["kind"]
                reg_date = attrs_dict["registrationDate"]

                if kind == "LINK":
                    path = ds.data["linkedData"]["contentCopies"][0]["path"]

                else:
                    path = ds.file_list[0]
                filename = path.rsplit("/", 1)[1]
                extension = filename.split(".")[-1].lower()
                local_filename = temp_dir + "/" + filename
                preview_format = "JPEG"

                if kind == "LINK":
                    size = ds.get_dataset_files().df.fileLength.sum()
                else:
                    size = ds.data["physicalData"]["size"]
                size = size / (1024**2)

                if include_preview and size < 500:

                    if kind == "LINK":
                        download_url = ds.p["s3_download_link"]
                    else:
                        base_url = "/".join(st.session_state.oBis.url.split("/")[:3])
                        token = st.session_state.oBis.token
                        openbis_filename = ds.file_list[0]
                        download_url = f'{base_url}/datastore_server/{permID}/{openbis_filename}?sessionID={token}'
                    
                    if download_url is not None:
                        try:
                            response = requests.get(download_url)
                            data_bytes = BytesIO(response.content)
                            if extension in ["jpg", "jpeg", "tiff", "tif", "tiff", "png", "bmp"]:
                                img = Image.open(data_bytes)
                            else:
                                with open(local_filename, "wb") as fh:
                                    fh.write(data_bytes.read())
                                if extension in VideoRenderer.ALLOWED_EXTENSIONS:
                                    if filename[4] == "-" and filename[10] == "T":
                                        name = filename.replace("_", "\n", 1)
                                        ds_type = dataset_type.replace("_", " ")
                                        name = name.replace(f"{dataset_type}_", f"{ds_type} ")
                                        name = name.replace("_", "\n", 1)
                                    else:
                                        name = filename
                                    preview_format = "GIF"
                                    renderer = VideoRenderer()
                                    img = renderer.get_image(path=local_filename, name=name)
                                elif dataset_type in preview_dict:
                                    # Previews are generated elsewhere (NOT on the spot)
                                    cache_client = st.session_state.s3_clients["S3_OPENBIS_CACHE"]
                                    cache_bucket = st.session_state.s3_bucket_names["S3_OPENBIS_CACHE"]
                                    try:
                                        prefix = preview_dict[dataset_type]
                                        response = cache_client.get_object(Bucket=cache_bucket, Key=f"preview/{prefix}/{permID}.jpg")
                                        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                                            img_data = response.get('Body').read()
                                            img = Image.open(BytesIO(img_data))
                                        else:
                                            img = Image.open("media/preview.jpg")
                                    except Exception as e:
                                        img = Image.open("media/preview.jpg")
                                elif dataset_type == "TEM_DATA":
                                    if extension == "emd":
                                        img_array = get_tem_stack(local_filename, False)
                                    elif extension in ["dm3", "dm4"]:
                                        img_array = get_tem_stack(local_filename, False)
                                    else:
                                        img_array = np.array(Image.open("media/preview.jpg"))
                                    img = Image.fromarray(img_array)
                                else:
                                    img = Image.open("media/preview.jpg")
                            preview = generate_preview(img, format=preview_format)
                        except Exception as e: # IOError,UnidentifiedImageError
                            preview = ""
                            st.toast(e)
                    else:
                        preview = ""


                    if os.path.exists(local_filename):
                        try:
                            os.unlink(local_filename)
                        except PermissionError:
                            pass

                else:
                    preview = ""

                if kind == "LINK":
                    path = ds.data["linkedData"]["contentCopies"][0]["path"]
                    size = ds.get_dataset_files().df.fileLength.sum()
                else:
                    path = None
                    size = ds.data["physicalData"]["size"]
                size = size / (1024**2)
                
                try:
                    datetime.strptime(filename[:26], "%Y-%m-%dT%H-%M-%S.%f")
                    filename = filename[27:].replace(f"{dataset_type}_", "")
                    filename = "_".join(filename.split("_")[1:])
                except ValueError:
                    pass
                entry = [preview, name, dataset_type, kind, reg_date, permID, filename, extension, path, size]
                data_all.append(entry + [ds.p[k] for k in md_keys])
                for opt in options:
                    entry.append(ds.p[opt])
                data.append(entry)
            st.session_state.df_files = pd.DataFrame(data, columns=cols)
            st.session_state.df_all = pd.DataFrame(data_all, columns=cols_all)
        if len(st.session_state.df_files) == 0:
            st.warning("No datasets!")

placeholder = st.empty()

with placeholder.form("GetData"):

    if (
        st.session_state.df_files is not None
        and len(st.session_state.df_files)
        and st.session_state.experiment is not None
    ):

        selection = dataframe_with_selections(st.session_state.df_files)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            download_btn = st.form_submit_button(
                label="Download selection",
                type="primary",
            )
        with col2:
            download_all_btn = st.form_submit_button(
                label=f"Download all ({len(st.session_state.df_files)})",
                type="primary",
                help="Avoid with large files",
            )
        with col3:
            delete_btn = st.form_submit_button(
                label="Delete selection",
                type="primary",
            )
        with col4:
            delete_all_btn = st.form_submit_button(
                label=f"Delete all ({len(st.session_state.df_files)})",
                type="primary",
            )

        if download_btn or download_all_btn:

            st.session_state.selection_deletion = pd.DataFrame()

            if download_btn:
                st.write("Your selection:")
                st.write(selection)
            if download_all_btn:
                selection = st.session_state.df_files
            st.session_state.selection = selection
            size_mb = selection["Size (Mb)"].sum()
            
            if size_mb > st.session_state.max_size:
                st.error(
                    f"Too much data to download ({size_mb:.2f} Gb > {st.session_state.max_size})   \n \
                        Please download files one by one"
                )
            else:
                oBis = st.session_state.oBis
                s3_clients = st.session_state.s3_clients.items()
                s3_bucket_names = st.session_state.s3_bucket_names
                if len(s3_bucket_names) == 0:
                    st.warning("You are not allowed to download files from Coscine")
                
                progress_bar = st.progress(0, text="Preparing files for download ...")
                
                for index, row in selection.reset_index().iterrows():
                    if len(selection) > 1:
                        progress_bar.progress(index/(len(selection)-1))
                    placeholder = st.empty()

                    if row["Kind"] == "PHYSICAL":
                        ds = oBis.get_dataset(row["permID"])
                        file_list = ds.file_list
                        for path in file_list:
                            filename = path.rsplit("/", 1)[1]
                            files_to_download.append(filename)
                        ds.download(
                            destination=temp_dir, create_default_folders=False
                        )
                    elif row["Kind"] == "LINK":
                        s3_filename = row["S3 Path"].rsplit("/", 1)[1]
                        dms_df = oBis.get_external_data_management_systems().df
                        dms_df = dms_df.set_index("code")
                        bucket_name, s3_client = None, None
                        for dmscode, s3_client in s3_clients:
                            if dms_df["address"].loc[dmscode] in row["S3 Path"]:
                                bucket_name = s3_bucket_names[dmscode]
                                break
                        if bucket_name is None:
                            placeholder.warning(
                                f"Skipping {s3_filename}",
                            )
                            continue
                        if s3_file_exists(
                            filename=s3_filename,
                            bucket=bucket_name,
                            s3_client=s3_client,
                        ):
                            placeholder.success(
                                f"File {s3_filename} exists in Coscine",
                                icon="✅",
                            )
                            files_to_download.append(s3_filename)
                            with st.spinner("Getting file from S3"):
                                s3_client.download_file(
                                    Bucket=bucket_name,
                                    Key=s3_filename,
                                    Filename=temp_dir + "/" + s3_filename,
                                )
                            # time.sleep(1)

                            placeholder.empty()
                        else:
                            placeholder.error(
                                "File not found in Coscine", icon="🔥"
                            )
                progress_bar.empty()    
        if delete_btn or delete_all_btn:
            st.session_state.requested_deletion = True
            if delete_all_btn:
                st.session_state.selection_deletion = st.session_state.df_files
            else:
                st.session_state.selection_deletion = selection
            
        # Dry run
        if len(st.session_state.selection_deletion):
            warning_messages = []
            for index, row in st.session_state.selection_deletion.iterrows(): 
                perm_id = row["permID"]
                ds = st.session_state.oBis.get_dataset(perm_id)
                # Design choice: only the person who uploaded the file can delete it in the GUI
                if st.session_state.openbis_username != ds.registrator:
                    warning_messages.append(
                        f"You're not allowed to delete this dataset: {perm_id}, registered by {ds.registrator}"
                    )
                    continue
                # We check if the s3 client is able to delete the file with the credentials given at login
                if row["Kind"] == "LINK":
                    s3_filename = row["S3 Path"].rsplit("/", 1)[1]
                    s3_client = st.session_state.s3_client
                    bucket_name = st.session_state.s3_bucket_name
                    dms_code = st.session_state.obis_dmscode
                    if s3_file_exists(
                        filename=s3_filename,
                        bucket=bucket_name,
                        s3_client=s3_client,
                    ):
                        try:
                            key = "dummy_file"
                            dummy_path = st.session_state.temp_dir + "/" + key
                            with open(dummy_path, "w") as f:
                                f.write("This is a test file.\n" * 1000)                        
                            s3_file_upload(dummy_path, bucket_name, key, s3_client)
                            s3_client.delete_object(
                                Bucket=bucket_name,
                                Key=key
                            )
                            os.path.unlink(dummy_path)
                        except Exception as e:
                            if "AccessDenied" in str(e):
                                warning_messages.append(
                                    f"You're not allowed to delete this file: {s3_filename}  \n" + \
                                    f"Access crendentials do not allow the DeleteObject operation  \n" + \
                                    "Are you using the right Coscine config file?"
                                )
                    else:
                        warning_messages.append(
                            f"You're not allowed to delete this file: {s3_filename}  \n" + \
                            f"Client cannot find file in bucket {bucket_name} registered with DMS code {dms_code}  \n" + \
                            "Are you using the right Coscine config file?"
                        )
            if len(warning_messages):
                st.warning("  \n".join(warning_messages))
            st.write("Your selection:")
            st.write(st.session_state.selection_deletion)
        
        if st.session_state.requested_deletion and len(st.session_state.selection_deletion):
            confirm_del_btn = st.form_submit_button(
                label="Confirm deletion",
                type="primary",
                disabled=not st.session_state.requested_deletion
            )
        else:
            st.session_state.requested_deletion = False
        
        # Delete files upon confirmation
        if st.session_state.requested_deletion and confirm_del_btn:
            with st.spinner("Deleting ..."):
                for index, row in st.session_state.selection_deletion.iterrows(): 
                    perm_id = row["permID"]
                    ds = st.session_state.oBis.get_dataset(perm_id)
                    
                    # Design choice: only the person who uploaded the file can delete it in the GUI
                    if st.session_state.openbis_username != ds.registrator:
                        continue
                    
                    # We check if the s3 client is able to delete the file with the credentials given at login
                    if row["Kind"] == "LINK":
                        s3_filename = row["S3 Path"].rsplit("/", 1)[1]
                        s3_client = st.session_state.s3_client
                        bucket_name = st.session_state.s3_bucket_name
                        dms_code = st.session_state.obis_dmscode

                        if s3_file_exists(
                            filename=s3_filename,
                            bucket=bucket_name,
                            s3_client=s3_client,
                        ):
                            try:
                                s3_client.delete_object(
                                    Bucket=bucket_name,
                                    Key=s3_filename
                                )
                                ds.delete(reason="Requested deletion through companion app.")
                            except Exception as e:
                                continue
                    
                    # For data stored on server
                    if row["Kind"] == "PHYSICAL":
                        ds.delete(reason="Requested deletion through companion app.")

            st.session_state.requested_deletion = False
            placeholder.empty()


# maybe a more elaborate logic with python tempdirs
# it seems to work at the moment - but keep in mind

if len(files_to_download) > 1:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    zip_name = "openBIS_download_" + timestamp + ".zip"
    st.session_state.df_all.drop("Preview", axis=1).to_csv(temp_dir + "/metadata.csv", index=False, encoding="utf-8-sig")
    with st.spinner("Creating zip archive"):
        with ZipFile(temp_dir + "/" + zip_name, "w", ZIP_DEFLATED) as zf:
            for file in files_to_download:
                zf.write(
                    filename=temp_dir + "/" + file,
                    arcname=file,
                )
            zf.write(temp_dir + "/metadata.csv", arcname="metadata.csv")
    with open(temp_dir + "/" + zip_name, "rb") as f:
        download_btn = st.download_button(
            label="Download zip archive",
            data=f,
            file_name=zip_name,
        )
elif len(files_to_download) == 1:
    with open(temp_dir + "/" + files_to_download[0], "rb") as f:
        download_btn = st.download_button(
            label="Download file",
            data=f,
            file_name=files_to_download[0],
        )

   
#
# cleanup
#


for file in os.scandir(temp_dir):
    if file.is_file():
        try:
            os.unlink(file)
        except PermissionError:
            pass
