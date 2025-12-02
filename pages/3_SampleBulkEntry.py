import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
import warnings

from Hello import find_relevant_locations


warnings.filterwarnings(action='ignore', category=FutureWarning)


st.set_page_config(
    page_title="Enter samples",
    page_icon="media/SFB1394_icon.jpg",
    layout="wide",
)

try:
    _ = st.session_state.setup_done
except AttributeError as e:
    st.switch_page("Hello.py")
st.sidebar.image("media/SFB1394_TitleImage_Cropped.png")
st.sidebar.write("Logged into openBIS: ", st.session_state.logged_in)
st.sidebar.write("OpenBIS Upload OK: ", st.session_state.openbis_upload_allowed)
st.sidebar.write("S3 Upload OK: ", st.session_state.s3_upload_ok)
st.sidebar.write("S3 Download OK: ", st.session_state.s3_download_ok)

st.title("Enter Samples from spreadsheet")


def perform_validation(df):
    COMPULSORY_COLUMNS = [
        "Name",
        "Location",
        "Sample Dimensions",
        "Date",
        "Element 1",
        "% Element 1",
    ]
    ATOMIC_SYMBOLS = [
        'H','He',
        'Li','Be','B','C','N','O','F','Ne',
        'Na','Mg','Al','Si','P','S','Cl','Ar',
        'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
        'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
        'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
        'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'
    ]

    detected_columns = set(df.columns.to_list())
    template = pd.read_excel("media/OpenBISSampleEntryTemplate.ods", engine="odf")
    allowed_columns = set(template.columns.to_list())

    if not detected_columns.issubset(set(allowed_columns)):
        not_expected = detected_columns.difference(set(allowed_columns))
        msg = f"**Columns do not match!**  \n"
        msg += f"**Expected**: {', '.join(template.columns.to_list())}  \n"
        msg += f"**Found**: {not_expected}"
        st.error(msg)
        return False
    if len(df) < 1:
        st.error("No sample detected")
        return False
    if df[COMPULSORY_COLUMNS].isnull().values.any():
        st.error(f"One of **{', '.join(COMPULSORY_COLUMNS)}** is empty.")
        return False
    element_cols = [
        col for col in detected_columns if "Element" in col and "%" not in col
    ]
    percent_cols = [col for col in detected_columns if "% Element" in col]
    for element in set(df[element_cols].dropna(axis=1).values.flatten()):
        if element not in ATOMIC_SYMBOLS:
            st.error(
                f"*{element}* is not an atomic symbol. Please use **atomic symbols**!"
            )
            return False
    is_valid_total = lambda x: x > 99.999 and x < 100.001
    if not df[percent_cols].dropna(axis=1).sum(axis=1).apply(is_valid_total).all():
        st.error("Make sure percentages add up to **100%**")
        return False
    is_valid_location = (
        lambda x: x.startswith("RWTH")
        or x.startswith("MPIE")
        or x.startswith("RUB")
        or x == "processed"
        or x == "virtual"
    )
    if not df["Location"].apply(is_valid_location).all():
        msg = "Location should be **processed** or **virtual**  \n"
        msg += "or **ORGANIZATION**-GROUP\:ROOMNUMBER,  \n"
        msg += "**ORGANIZATION** must be one of the following: **RWTH,RUB,MPIE**"
        st.error(msg)
        return False
    is_valid_date = lambda x: len(x) == 10 and x[4] == "-" and x[7] == "-"
    if not df["Date"].apply(is_valid_date).all():
        msg = "**Date** should be **YYYY-MM-DD**\n"
        st.error(msg)
        return False
    terms_crystal = {"MONOCRYSTALLINE", "BICRYSTALLINE", "POLYCRYSTALLINE"}
    not_null_values = set(df["Crystal Type"].dropna().unique())
    if not df["Crystal Type"].isna().all() and not not_null_values.issubset(
        terms_crystal
    ):
        msg = "**Crystal Type** should be one of the following:  \n"
        msg += f"{', '.join(terms_crystal)}"
        st.error(msg)
        return False
    object_permids = []
    for col_name in ["Parents", "Children"]:
        col = df[col_name].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
        col = col.str.replace(", ", ",").str.split(",")
        object_permids.extend(col.explode().tolist())
    object_permids = set(object_permids)
    object_permids.remove("")

    for perm_id in set(object_permids):
        try:
            st.session_state.oBis.get_object(perm_id)
        except ValueError:
            st.error(
                f"**Parent/Child** Object with permId = {perm_id} does **NOT** exist!"
            )
            return False
    st.success("Valid spreadsheet")
    return True


st.warning("Please use atomic symbols and atomic percentages!")
st.warning(
    "Please enter Parents and Children as comma-seperated permIds or identifiers"
)
download_btn = st.download_button(
    label="Download template file",
    data=open("media/OpenBISSampleEntryTemplate.ods", "rb"),
    file_name="OpenBISSampleEntryTemplate.ods",
)

spreadsheet = st.file_uploader(
    "Upload filled spreadsheet (do no modify the columns)",
    accept_multiple_files=False,
    type=["ods"],
    disabled=not st.session_state.setup_done,
)

st.session_state.table_loaded = False
if spreadsheet:

    df = pd.read_excel(spreadsheet, engine="odf")

    if perform_validation(df):
        for col_name in ["Parents", "Children"]:
            df[col_name] = df[col_name].fillna("")
            df[col_name] = df[col_name].str.strip()
            df[col_name] = df[col_name].str.replace(r"\s+", " ", regex=True)
            df[col_name] = df[col_name].str.replace(", ", ",")
            df[col_name] = df[col_name].str.split(",")
        
        ot = st.session_state.oBis.get_object_type("SAMPLE")
        mapper = dict()
        for assignment in ot.get_property_assignments():
            mapper[assignment.label] = assignment.code.lower()
        df = df.rename(columns=mapper)
        df = df.astype(object).where(pd.notnull(df), None)
        df["composition_desc"] = "ATOMIC_FRACTION"
        st.session_state.table_loaded = True
        st.session_state.samples_df = df
with st.form("EnterSamples"):

    refresh_btn = st.form_submit_button(
        "Refresh",
        type="primary",
        disabled=not (st.session_state.setup_done and st.session_state.table_loaded),
        help="Click if you added ELN entries after login",
    )

    if refresh_btn:
        with st.spinner("Please wait ..."):
            _ = find_relevant_locations(
                username=st.session_state.openbis_username,
                include_samples=st.session_state.include_samples,
            )

    collections_of_samples = dict()
    for display_name, identifier in st.session_state.experiments.items():
        if identifier.count("/") == 3:
            collection = st.session_state.oBis.get_experiment(identifier)
            if (
                collection.type == "COLLECTION"
                and collection.p.get("$default_object_type") == "SAMPLE"
            ):
                collections_of_samples[display_name] = identifier
    st.selectbox(
        "Choose in which Collection your samples will be created",
        list(collections_of_samples.keys()),
        index=None,
        placeholder="Select collection",
        key="collection",
        disabled=not st.session_state.table_loaded,
    )

    confirm_btn = st.form_submit_button(
        "Create Samples in OpenBIS",
        type="primary",
        disabled=not st.session_state.table_loaded,
    )

    if confirm_btn:
        collection_name = collections_of_samples[st.session_state.collection]

        try:
            st.session_state.oBis.get_collection(collection_name)
        except ValueError:
            st.error("Collection not found!")
            collection_name = None
        if collection_name:
            progress_text = "Creating samples. Please wait."
            progress_bar = st.progress(0, text=progress_text)

            df = st.session_state.samples_df

            new_permids = []

            for i, row in df.iterrows():
                progress_bar.progress((i + 1) / len(df), text=progress_text)
                row = row.dropna()
                props = row.to_dict()
                parents = props.pop("Parents", [])
                children = props.pop("Children", [])
                if "" in parents:
                    parents.remove("")
                if "" in children:
                    children.remove("")
                
                new_sample = st.session_state.oBis.new_object(
                    type="SAMPLE",
                    props=props,
                    parents=parents,
                    children=children,
                    experiment=collection_name,
                )
                new_sample.save()
                permid = new_sample.permId
                new_permids.append(permid)
            progress_bar.empty()
            st.success(
                f"Samples added succesfully. \n",
                icon="✅",
            )
            st.text(f'PermIds: {", ".join(new_permids)}')
            st.session_state.table_loaded = False
            st.session_state.samples_df = None
