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


from s3_tools import s3_file_exists
from pybis_tools import get_info_from_identifier

# Utility to update experiment list


from Hello import find_relevant_locations

# Utilities for preview
from metadata.metadata_Velox import get_image_stack as get_tem_stack_from_emd
from metadata.metadata_GMS import get_image_stack as get_tem_stack_from_dm

warnings.filterwarnings(action='ignore', category=FutureWarning)

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


def generate_preview(pil_img: Image, img_height: int = 320):
    w, h = pil_img.size
    img = np.array(pil_img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-20)
    img = (img * 255).astype("uint8")
    img = Image.fromarray(img)
    img = img.resize(
        size=(int(img_height*w/h), img_height),
        resample=Image.Resampling.BOX,
    )
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    preview = f"data:image/jpeg;base64,{img_base64}"
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
    st.session_state.requested_deletion = False
except AttributeError as e:
    st.switch_page("Hello.py")
st.sidebar.image("media/SFB1394_TitleImage_Cropped.png")
st.sidebar.write("Logged into openBIS: ", st.session_state.logged_in)
st.sidebar.write("OpenBIS Upload OK: ", st.session_state.openbis_upload_allowed)
st.sidebar.write("S3 Upload OK: ", st.session_state.s3_upload_ok)
st.sidebar.write("S3 Download OK: ", st.session_state.s3_download_ok)


temp_dir = st.session_state.temp_dir
if "df_files" not in st.session_state:
    st.session_state.df_files = None
st.title("Show Linked Data in openBIS")
st.write(
    "Note: If you select multiple files to download, "
    + "they will be compressed into a zip archive."
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
    # Prompt users to choose an object otr an experiment/collection

    st.selectbox(
        "Find where the data you are looking for is located",
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
        with st.spinner(text="Preparing files ..."):
            identifier = st.session_state.experiments[st.session_state.experiment]
            exp_name, object_name = get_info_from_identifier(identifier)

            datasets = st.session_state.oBis.get_datasets(
                experiment=exp_name,
                sample=object_name,
            )
            md_keys = set()

            for ds in datasets:
                md_keys.update(ds.p.all().keys())
            if "$name" in md_keys:
                md_keys.remove("$name")  # Already in dataframe
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
                "S3 Path",
                "Size (Mb)",
            ]

            for opt in options:
                opt = st.session_state.oBis.get_property_type(opt).label
                cols.append(opt)
            
            if datasets is not None and include_preview:
                n_files = len(datasets.df)
                if n_files > 500:
                    st.warning(f"Skipping preview, too many files ({n_files})")
                    include_preview = False
            
            data = []
            for ds in datasets:
                name = ds.p["$name"]
                permID = ds.permId
                attrs_dict = ds.attrs.all()
                type = attrs_dict["type"]
                kind = attrs_dict["kind"]
                reg_date = attrs_dict["registrationDate"]

                if kind == "LINK":
                    path = ds.data["linkedData"]["contentCopies"][0]["path"]
                else:
                    path = ds.file_list[0]
                filename = path.rsplit("/", 1)[1]
                
                if include_preview:
                    if kind == "LINK":
                        local_filename = temp_dir + '/' + filename
                        extension = filename.split('.')[-1].lower()
                        s3_url = ds.p['s3_download_link']
                        if s3_url is not None:
                            try:
                                response = requests.get(s3_url)
                                data_bytes = BytesIO(response.content)
                                if extension in ['jpg', 'jpeg', 'tiff', 'tif', 'tiff', 'png', 'bmp']:
                                    img = Image.open(data_bytes)
                                elif type == 'TEM_DATA':
                                    with open(local_filename, 'wb') as fh:
                                        fh.write(data_bytes.read())
                                    if extension == 'emd':
                                        img_array = get_tem_stack_from_emd(local_filename, False)
                                    elif extension in ['dm3', 'dm4']:
                                        img_array = get_tem_stack_from_dm(local_filename, False)
                                    else:
                                        img_array = np.array(Image.open('media/preview.jpg'))
                                    img = Image.fromarray(img_array)
                                else:
                                    img = Image.open('media/preview.jpg')
                                preview = generate_preview(img)
                            except Exception as e: # IOError,UnidentifiedImageError
                                preview = ""
                        else:
                            preview = ""

                    else:
                        
                        ds.download(
                            files={path},
                            destination=temp_dir,
                            create_default_folders=False,
                            wait_until_finished=True,
                        )

                        local_filename = temp_dir + '/' + filename
                        extension = filename.split('.')[-1].lower()
                        try:
                            if extension in ['jpg', 'jpeg', 'tiff', 'tif', 'tiff', 'png', 'bmp']:
                                img = Image.open(local_filename)
                            elif type == 'TEM_DATA':
                                if extension == 'emd':
                                    img_array = get_tem_stack_from_emd(local_filename, False)
                                elif extension in ['dm3', 'dm4']:
                                    img_array = get_tem_stack_from_dm(local_filename, False)
                                else:
                                    img_array = np.array(Image.open('media/preview.jpg'))
                                img = Image.fromarray(img_array)
                            else:
                                img = Image.open('media/preview.jpg')
                            preview = generate_preview(img)
                        except Exception as e: # IOError,UnidentifiedImageError
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
                    filename = filename[27:].replace(f"{type}_", "")
                    filename = "_".join(filename.split("_")[1:])
                except ValueError:
                    pass
                entry = [preview, name, type, kind, reg_date, permID, filename, path, size]
                for opt in options:
                    entry.append(ds.p[opt])
                data.append(entry)
            st.session_state.df_files = pd.DataFrame(data, columns=cols)
        if len(st.session_state.df_files) == 0:
            st.warning("No datasets!")

files_to_download = []
zip_name = "openBIS_download_" + datetime.now().isoformat("T") + ".zip"
zip_name = zip_name.replace(":", "-")

with st.form("GetData"):

    if (
        st.session_state.df_files is not None
        and len(st.session_state.df_files)
        and st.session_state.experiment is not None
    ):

        selection = dataframe_with_selections(st.session_state.df_files)
        col1, col2, col3 = st.columns(3)
        with col1:
            download_btn = st.form_submit_button(
                f"Download selection",
                type="primary",
            )
        with col2:
            download_all_btn = st.form_submit_button(
                f"Download all ({len(st.session_state.df_files)})",
                type="primary",
                help="Avoid with large files",
            )
        with col3:
            delete_btn = st.form_submit_button(
                f"Delete selection",
                type="primary",
            )
        if download_btn or download_all_btn:
            if download_btn:
                st.write("Your selection:")
                st.write(selection)
            if download_all_btn:
                selection = st.session_state.df_files
            size_mb = selection["Size (Mb)"].sum()
            
            if size_mb  > st.session_state.max_size:
                st.error(
                    f"Too much data to download ({size_mb:.2f} Gb > {st.session_state.max_size})   \n \
                         Please download files one by one"
                )
            else:
                with st.spinner("Preparing files for download ..."):
                    oBis = st.session_state.oBis
                    s3_clients = st.session_state.s3_clients.items()
                    s3_bucket_names = st.session_state.s3_bucket_names

                    if len(s3_bucket_names) == 0:
                        st.warning("You are not allowed to download files from Coscine")
                    for index, row in selection.iterrows():

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
        if delete_btn:
            st.session_state.requested_deletion = True
            st.write("Your selection:")
            st.write(selection)
        else:
            st.session_state.requested_deletion = False


# maybe a more elaborate logic with python tempdirs
# it seems to work at the moment - but keep in mind

if st.session_state.requested_deletion:
    with st.form("DeleteFiles"):
        confirm_del_btn = st.form_submit_button(
            label="Confirm deletion",
            type="primary"
        )
        for index, row in selection.iterrows():
            perm_id = row["permID"]
            ds = st.session_state.oBis.get_dataset(perm_id)
            if st.session_state.openbis_username != ds.registrator:
                st.warning(
                    f"You're not allowed to delete this dataset: {perm_id}, registered by {ds.registrator}"
                )
                continue
            if row["Kind"] == "LINK":
                s3_filename = row["S3 Path"].rsplit("/", 1)[1]
                s3_client = st.session_state.s3_client
                bucket_name = st.session_state.s3_bucket_name

                if s3_file_exists(
                    filename=s3_filename,
                    bucket=bucket_name,
                    s3_client=s3_client,
                ):
                    s3_client.delete_object(
                        Bucket=bucket_name,
                        Key=s3_filename
                    )
                    ds.delete(reason="Requested deletion through companion app.")
                else:
                    st.warning(f"You're not allowed to delete this file: {s3_filename}")
            elif row["Kind"] == "PHYSICAL":
                ds.delete(reason="Requested deletion through companion app.")


if len(files_to_download) > 1:
    with st.spinner("Creating zip archive"):
        with ZipFile(temp_dir + "/" + zip_name, "w", ZIP_DEFLATED) as zf:
            for file in files_to_download:
                zf.write(
                    filename=temp_dir + "/" + file,
                    arcname=file,
                )
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
    if file.is_file:
        try:
            os.unlink(file)
        except PermissionError:
            pass
