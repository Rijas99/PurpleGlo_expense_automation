import streamlit as st
import pandas as pd
import os
import shutil
import json
import stat
import re
import time as time_module
import hashlib
from datetime import datetime, time as dt_time
from PIL import Image
import google.generativeai as genai

# =========================================================
# CONFIGURATION
# =========================================================

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

WORK_DIR = "expense_workspace"
CURRENT_DIR = os.path.join(WORK_DIR, "current")
HISTORY_DIR = os.path.join(WORK_DIR, "history")

IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
DATA_FILE = os.path.join(CURRENT_DIR, "data.csv")
CREDIT_CARD_DATA_FILE = os.path.join(CURRENT_DIR, "credit_card_data.csv")
TRANSPORT_DATA_FILE = os.path.join(CURRENT_DIR, "transport_data.csv")

MAX_HISTORY_MONTHS = 3  # ✅ keep only last 3 months

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def parse_retry_seconds(err_msg: str) -> int | None:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_msg)
    if m:
        return int(m.group(1))
    return None


def _remove_readonly(func, path, _excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def safe_rmtree(path, retries=5, delay=0.5):
    if not os.path.exists(path):
        return
    last_err = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_remove_readonly)
            return
        except PermissionError as e:
            last_err = e
            if attempt < retries - 1:
                time_module.sleep(delay)
            else:
                raise PermissionError(
                    f"Cannot delete '{path}'. A file/folder is open.\n"
                    f"Close File Explorer windows / Excel / Zip preview and try again."
                ) from e
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time_module.sleep(delay)
    if last_err:
        raise last_err


def ensure_current_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def month_folder_path(month_slug: str):
    return os.path.join(HISTORY_DIR, month_slug)


def list_archived_months():
    if not os.path.exists(HISTORY_DIR):
        return []
    months = []
    for name in os.listdir(HISTORY_DIR):
        p = os.path.join(HISTORY_DIR, name)
        if os.path.isdir(p):
            months.append(name)
    # sort by folder modified time (latest first)
    months.sort(
        key=lambda m: os.path.getmtime(os.path.join(HISTORY_DIR, m)), reverse=True
    )
    return months


def trim_history_keep_last_n(n: int):
    months = list_archived_months()
    if len(months) <= n:
        return
    to_delete = months[n:]
    for m in to_delete:
        safe_rmtree(os.path.join(HISTORY_DIR, m))


def archive_current_month(report_month: str):
    """
    Archive CURRENT_DIR into HISTORY_DIR/<month_slug>/ then reset CURRENT_DIR.
    Keep only last MAX_HISTORY_MONTHS archives.
    """
    ensure_current_dirs()
    month_slug = slugify(report_month)  # e.g. Jan_2026
    target = month_folder_path(month_slug)

    if os.path.exists(target):
        raise RuntimeError(
            f"Archive folder already exists: {month_slug}\n"
            f"Change Report Month or delete that archive folder."
        )

    # create target
    os.makedirs(target, exist_ok=True)
    os.makedirs(os.path.join(target, "images"), exist_ok=True)

    # copy csvs if exist
    if os.path.exists(DATA_FILE):
        shutil.copy(DATA_FILE, os.path.join(target, "data.csv"))
    if os.path.exists(CREDIT_CARD_DATA_FILE):
        shutil.copy(CREDIT_CARD_DATA_FILE, os.path.join(target, "credit_card_data.csv"))
    if os.path.exists(TRANSPORT_DATA_FILE):
        shutil.copy(TRANSPORT_DATA_FILE, os.path.join(target, "transport_data.csv"))

    # copy images
    if os.path.exists(IMAGES_DIR):
        shutil.copytree(IMAGES_DIR, os.path.join(target, "images"), dirs_exist_ok=True)

    # reset current
    safe_rmtree(CURRENT_DIR)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # ✅ keep last 3 only
    trim_history_keep_last_n(MAX_HISTORY_MONTHS)


def get_paths_for_month(selected_month_slug: str | None):
    if not selected_month_slug:  # CURRENT
        base = CURRENT_DIR
    else:
        base = os.path.join(HISTORY_DIR, selected_month_slug)

    images_dir = os.path.join(base, "images")
    data_file = os.path.join(base, "data.csv")
    cc_file = os.path.join(base, "credit_card_data.csv")
    transport_file = os.path.join(base, "transport_data.csv")
    return images_dir, data_file, cc_file, transport_file


def load_data_for(selected_month_slug: str | None):
    _, data_file, _, _ = get_paths_for_month(selected_month_slug)
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    return pd.DataFrame(
        columns=[
            "Ref",
            "Date",
            "Description",
            "Category",
            "Project Code",
            "Amount",
            "Original_Image_Path",
        ]
    )


def save_data_current(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)


def delete_receipt(ref: int) -> tuple[bool, str]:
    """
    Delete a receipt by its Ref number.
    Removes the row from the dataframe and deletes the associated image file.
    Returns (success: bool, message: str)
    """
    try:
        df = load_data_for(None)  # Load current month data

        # Find the row with matching Ref
        row_to_delete = df[df["Ref"] == ref]
        if row_to_delete.empty:
            return False, f"Receipt with Ref {ref} not found."

        # Get the image path before deleting the row
        image_path = None
        if "Original_Image_Path" in row_to_delete.columns:
            image_path = row_to_delete["Original_Image_Path"].iloc[0]

        # Remove the row from dataframe
        df = df[df["Ref"] != ref]

        # Delete the associated image file if it exists
        if image_path:
            full_image_path = os.path.join(IMAGES_DIR, str(image_path))
            if os.path.exists(full_image_path):
                try:
                    os.remove(full_image_path)
                except Exception as e:
                    # Log but don't fail if image deletion fails
                    pass

        # Save the updated dataframe
        save_data_current(df)
        return True, f"Receipt {ref} deleted successfully."

    except Exception as e:
        return False, f"Error deleting receipt: {str(e)}"


def load_cc_for(selected_month_slug: str | None):
    _, _, cc_file, _ = get_paths_for_month(selected_month_slug)
    if os.path.exists(cc_file):
        return pd.read_csv(cc_file)
    return pd.DataFrame(
        columns=["Date", "Description", "Category", "Project Code", "Amount"]
    )


def save_cc_current(df: pd.DataFrame):
    df.to_csv(CREDIT_CARD_DATA_FILE, index=False)


def load_transport_data(selected_month_slug: str | None = None):
    """Load transport data for current or archived month."""
    if not selected_month_slug:  # CURRENT
        transport_file = TRANSPORT_DATA_FILE
    else:
        transport_file = os.path.join(
            HISTORY_DIR, selected_month_slug, "transport_data.csv"
        )

    if os.path.exists(transport_file):
        return pd.read_csv(transport_file)
    return pd.DataFrame(
        columns=[
            "Date",
            "From",
            "Destination",
            "Return Included",
            "Project Code",
        ]
    )


def save_transport_data(df: pd.DataFrame):
    """Save transport data to current month CSV."""
    df.to_csv(TRANSPORT_DATA_FILE, index=False)


def generate_transport_excel(df: pd.DataFrame, output_path: str):
    """
    Generate Excel file for transport expenses.
    Columns: Date, From, Destination, Return Included, Project Code
    """
    if df.empty:
        st.warning("No transport expenses to export.")
        return

    # Create a new DataFrame with the required columns
    excel_df = pd.DataFrame()
    excel_df["Date"] = df["Date"]
    excel_df["From"] = df["From"]
    excel_df["Destination"] = df["Destination"]
    # Convert boolean to "Return Included" or empty string
    excel_df["Return Included"] = df["Return Included"].apply(
        lambda x: "Return Included" if x else ""
    )
    excel_df["Project Code"] = df["Project Code"]

    # Write to Excel
    excel_df.to_excel(output_path, index=False, engine="openpyxl")


def get_current_month_str(date_obj=None):
    if date_obj is None:
        date_obj = datetime.now()
    elif isinstance(date_obj, datetime):
        date_obj = date_obj.date()
    return date_obj.strftime("%b")


def get_project_codes_for_month(
    month_str=None, date_obj=None, include_credit_card=True
):
    if month_str is None:
        month_str = get_current_month_str(date_obj)

    all_codes = []
    df = load_data_for(None)  # only from CURRENT for code suggestions
    if not df.empty:
        month_filter = df["Date"].str.contains(f"-{month_str}$", case=False, na=False)
        month_data = df[month_filter]
        for code in month_data["Project Code"].dropna():
            code_str = str(code)
            if code_str.endswith(".0") and code_str.replace(".0", "").isdigit():
                code_str = code_str[:-2]
            if code_str.strip():
                all_codes.append(code_str)

    if include_credit_card:
        cc_df = load_cc_for(None)
        if not cc_df.empty:
            cc_month_filter = cc_df["Date"].str.contains(
                f"-{month_str}$", case=False, na=False
            )
            cc_month_data = cc_df[cc_month_filter]
            for code in cc_month_data["Project Code"].dropna():
                code_str = str(code)
                if code_str.endswith(".0") and code_str.replace(".0", "").isdigit():
                    code_str = code_str[:-2]
                if code_str.strip():
                    all_codes.append(code_str)

    return sorted(list(set(all_codes)))


def get_meal_description(time_obj):
    if isinstance(time_obj, datetime):
        time_obj = time_obj.time()
    elif not isinstance(time_obj, dt_time):
        if isinstance(time_obj, str):
            try:
                hour, minute = map(int, time_obj.split(":"))
                time_obj = dt_time(hour, minute)
            except:
                time_obj = datetime.now().time()
        else:
            time_obj = datetime.now().time()

    hour = time_obj.hour
    if 5 <= hour < 12:
        return "Breakfast"
    elif 12 <= hour < 18:
        return "Lunch"
    else:
        return "Dinner"


def _extract_json(text: str) -> dict:
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in Gemini response.")
        return json.loads(m.group(0))


# =========================================================
# GEMINI (button-based, cached, 429 retry once)
# =========================================================


def analyze_receipt_gemini(image_file):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not set. Add to .streamlit/secrets.toml or env var.")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_file)

    prompt = """
Analyze this receipt image. Extract strictly JSON:
1) date YYYY-MM-DD
2) time HH:MM (24h) or null
3) amount numeric only
4) description short
5) category one of:
["Hotel Booking","Food & Beverages","Visa & Ticket","Parking","R & D Expenses","Subscriptions",
"Office - Tools & Consumables","Project - Consumables","Transportation",
"Project Expenses - Miscellaneous","Office Expenses - Miscellaneous","Can't classify"]

Return ONLY JSON.
"""
    try:
        response = model.generate_content([prompt, img])
        return _extract_json(response.text)
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            wait_s = parse_retry_seconds(msg) or 30
            st.warning(f"Rate limit hit. Waiting {wait_s}s then retrying once...")
            time_module.sleep(wait_s)
            response = model.generate_content([prompt, img])
            return _extract_json(response.text)
        st.error(f"AI Error: {e}")
        return None


def analyze_credit_card_statement(image_file):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not set. Add to .streamlit/secrets.toml or env var.")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_file)

    prompt = """
Analyze this credit card statement image. Extract TOTAL expense as JSON:
{"date":"YYYY-MM-DD","description":"...","amount":123.45,"category":"..."}
Category must be one of:
["Hotel Booking","Food & Beverages","Visa & Ticket","Parking","R & D Expenses","Subscriptions",
"Office - Tools & Consumables","Project - Consumables","Transportation",
"Project Expenses - Miscellaneous","Office Expenses - Miscellaneous","Can't classify"]

Return ONLY JSON.
"""
    try:
        response = model.generate_content([prompt, img])
        return _extract_json(response.text)
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            wait_s = parse_retry_seconds(msg) or 30
            st.warning(f"Rate limit hit. Waiting {wait_s}s then retrying once...")
            time_module.sleep(wait_s)
            response = model.generate_content([prompt, img])
            return _extract_json(response.text)
        st.error(f"AI Error: {e}")
        return None


# =========================================================
# REPORT GENERATION (exports selected month: CURRENT or archived)
# =========================================================


def generate_receipts_package(report_month: str, selected_month_slug: str | None):
    images_dir, data_file, _, _ = get_paths_for_month(selected_month_slug)
    df = load_data_for(selected_month_slug)

    if df.empty:
        st.error("No receipts expenses found for this month selection.")
        return

    month_slug = selected_month_slug if selected_month_slug else slugify(report_month)

    output_folder = os.path.join(WORK_DIR, f"receipts_output_{month_slug}")
    bills_folder = os.path.join(output_folder, f"bills_{month_slug}")

    if os.path.exists(output_folder):
        safe_rmtree(output_folder)
    os.makedirs(bills_folder, exist_ok=True)

    # copy images
    if "Original_Image_Path" in df.columns:
        for _, row in df.iterrows():
            src = os.path.join(images_dir, str(row["Original_Image_Path"]))
            if os.path.exists(src):
                dst = os.path.join(bills_folder, f"{row['Ref']}.jpg")
                shutil.copy(src, dst)

    # excel export
    excel_path = os.path.join(output_folder, f"Receipts_Expenses_{month_slug}.xlsx")
    export_df = df.copy()
    if "Original_Image_Path" in export_df.columns:
        export_df = export_df.drop(columns=["Original_Image_Path"])
    export_df.to_excel(excel_path, index=False)

    zip_base = os.path.join(WORK_DIR, f"Receipts_Submission_{month_slug}")
    zip_path = f"{zip_base}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_base, "zip", output_folder)

    st.success("Receipts package generated!")
    with open(zip_path, "rb") as fp:
        st.download_button(
            "⬇️ Download Receipts ZIP",
            fp,
            file_name=f"Receipts_Submission_{month_slug}.zip",
            mime="application/zip",
        )


def generate_credit_card_package(report_month: str, selected_month_slug: str | None):
    cc_df = load_cc_for(selected_month_slug)
    if cc_df.empty:
        st.error("No credit card expenses found for this month selection.")
        return

    month_slug = selected_month_slug if selected_month_slug else slugify(report_month)

    output_folder = os.path.join(WORK_DIR, f"credit_output_{month_slug}")
    if os.path.exists(output_folder):
        safe_rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    excel_path = os.path.join(output_folder, f"CreditCard_Expenses_{month_slug}.xlsx")
    cc_df.to_excel(excel_path, index=False)

    zip_base = os.path.join(WORK_DIR, f"CreditCard_Submission_{month_slug}")
    zip_path = f"{zip_base}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_base, "zip", output_folder)

    st.success("Credit card package generated!")
    with open(zip_path, "rb") as fp:
        st.download_button(
            "⬇️ Download Credit Card ZIP",
            fp,
            file_name=f"CreditCard_Submission_{month_slug}.zip",
            mime="application/zip",
        )


# =========================================================
# UI
# =========================================================

st.set_page_config(
    page_title="PurpleGlo Expense Manager",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# Mobile-friendly CSS
st.markdown(
    """
    <style>
    /* Increase touch target sizes for mobile */
    .stButton > button {
        min-height: 48px;
        font-size: 16px;
    }
    /* Make inputs larger and easier to tap */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        font-size: 16px;
        padding: 12px;
    }
    /* Improve spacing on mobile */
    .element-container {
        margin-bottom: 1rem;
    }
    /* Make date picker more touch-friendly */
    .stDateInput > div > div > input {
        font-size: 16px;
        padding: 12px;
    }
    /* Mobile-friendly table container with horizontal scroll */
    .mobile-table-container {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        width: 100%;
        margin: 1rem 0;
    }
    .mobile-table-container table {
        width: 100%;
        min-width: 600px;
        border-collapse: collapse;
        font-size: 14px;
    }
    .mobile-table-container th {
        background-color: rgba(250, 250, 250, 0.1);
        padding: 12px 8px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        white-space: nowrap;
    }
    .mobile-table-container td {
        padding: 12px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        word-wrap: break-word;
    }
    .mobile-table-container tr:hover {
        background-color: rgba(255, 255, 255, 0.05);
    }
    /* Mobile-optimized delete button */
    .mobile-delete-btn {
        min-width: 48px;
        min-height: 48px;
        padding: 8px 12px;
        font-size: 16px;
    }
    /* Responsive column widths */
    @media (max-width: 768px) {
        .mobile-table-container table {
            font-size: 13px;
        }
        .mobile-table-container th,
        .mobile-table-container td {
            padding: 10px 6px;
        }
    }
    /* Ensure table is scrollable on mobile */
    .dataframe {
        overflow-x: auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🟣 PurpleGlo Expense Manager")

# Sidebar
st.sidebar.header("Settings")
report_month = st.sidebar.text_input(
    "Report Month", value=datetime.now().strftime("%b %Y")
)
st.sidebar.write("Gemini API key loaded:", bool(GOOGLE_API_KEY))

st.sidebar.divider()

archived = list_archived_months()
month_view = st.sidebar.selectbox("View Month", options=["CURRENT"] + archived, index=0)
selected_month_slug = None if month_view == "CURRENT" else month_view

if month_view != "CURRENT":
    st.sidebar.info(f"Viewing archived month: {month_view}")

if st.sidebar.button("🗑️ Start New Month (Archive + Clear)"):
    try:
        archive_current_month(report_month)
        st.success("Archived current month and started a new month.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

tab1, tab2, tab3 = st.tabs(["📸 Add Receipt", "💳 Credit Card", "🚗 Transportation"])

# =========================================================
# TAB 1: RECEIPTS
# =========================================================
with tab1:
    st.subheader("Receipts")

    if month_view != "CURRENT":
        st.warning(
            "You are viewing archived data (read-only). Switch to CURRENT to add new receipts."
        )

    df_view = load_data_for(selected_month_slug)
    if not df_view.empty:
        # Prepare display dataframe with optimized column widths for mobile
        display_df = df_view[
            ["Ref", "Date", "Description", "Category", "Project Code", "Amount"]
        ].copy()

        # Format and truncate for mobile display
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"{x:.2f}")
        display_df["Description"] = display_df["Description"].apply(
            lambda x: (str(x)[:35] + "...") if len(str(x)) > 35 else str(x)
        )
        display_df["Category"] = display_df["Category"].apply(
            lambda x: (str(x)[:22] + "...") if len(str(x)) > 22 else str(x)
        )
        display_df["Project Code"] = display_df["Project Code"].apply(
            lambda x: (str(x)[:18] + "...") if len(str(x)) > 18 else str(x)
        )

        # Display scrollable table - st.dataframe handles mobile scrolling well
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ref": st.column_config.NumberColumn("Ref", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Description": st.column_config.TextColumn(
                    "Description", width="medium"
                ),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Project Code": st.column_config.TextColumn(
                    "Project Code", width="medium"
                ),
                "Amount": st.column_config.NumberColumn(
                    "Amount", width="small", format="%.2f"
                ),
            },
        )

        # Delete buttons section - mobile-friendly grid layout
        if month_view == "CURRENT":
            st.divider()
            st.write("**🗑️ Delete Receipts:**")
            # Create responsive grid: 2 columns on mobile, 3-4 on larger screens
            num_cols = (
                min(3, len(df_view)) if len(df_view) > 3 else max(2, len(df_view))
            )
            delete_cols = st.columns(num_cols)

            for idx, row in df_view.iterrows():
                col_idx = idx % num_cols
                ref_value = int(row["Ref"])
                with delete_cols[col_idx]:
                    if st.button(
                        f"🗑️ Delete #{ref_value}",
                        key=f"delete_receipt_{ref_value}",
                        use_container_width=True,
                        help=f"Delete receipt {ref_value}: {row['Description'][:25]}",
                    ):
                        success, message = delete_receipt(ref_value)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        st.metric("Total Receipts", f"{df_view['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No receipt expenses for this month selection.")

    if month_view == "CURRENT":
        uploaded_file = st.file_uploader(
            "Upload Receipt (Camera or File)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=False,
            help="📷 Tap to take a photo with your camera or browse files",
            key="receipt_upload",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Receipt Preview", use_container_width=True)
            h = file_hash(uploaded_file)

            analyze_btn = st.button(
                "🤖 Analyze Receipt with AI",
                key="rcpt_analyze_btn",
                use_container_width=True,
            )
            clear_btn = st.button(
                "♻️ Clear AI Result", key="rcpt_clear_btn", use_container_width=True
            )

            if clear_btn:
                st.session_state.pop("receipt_ai_hash", None)
                st.session_state.pop("receipt_ai_data", None)

            if analyze_btn:
                if st.session_state.get(
                    "receipt_ai_hash"
                ) == h and st.session_state.get("receipt_ai_data"):
                    st.info("Using cached AI result.")
                else:
                    with st.spinner("Reading receipt with Gemini..."):
                        ai_data = analyze_receipt_gemini(uploaded_file)
                        if ai_data:
                            st.session_state["receipt_ai_data"] = ai_data
                            st.session_state["receipt_ai_hash"] = h
                            st.toast("Receipt AI completed!")

            ai_data = st.session_state.get("receipt_ai_data", {})
            default_date = datetime.today()
            default_desc = ""
            default_cat = "Project - Consumables"
            default_amt = 0.0
            default_time = None

            if ai_data:
                try:
                    default_date = datetime.strptime(ai_data.get("date"), "%Y-%m-%d")
                except:
                    pass
                default_time = ai_data.get("time")
                default_cat = ai_data.get("category", default_cat)
                try:
                    default_amt = float(ai_data.get("amount", default_amt))
                except:
                    pass

                if default_cat == "Food & Beverages":
                    default_desc = get_meal_description(
                        default_time if default_time else datetime.now()
                    )
                else:
                    default_desc = ai_data.get("description", "")

            with st.form("receipt_form"):
                # Use single column on mobile for better touch experience
                c1, c2 = st.columns([1, 1])
                with c1:
                    date_input = st.date_input("Date", value=default_date)
                    formatted_date = date_input.strftime("%d-%b")

                    category_options = [
                        "Hotel Booking",
                        "Food & Beverages",
                        "Visa & Ticket",
                        "Parking",
                        "R & D Expenses",
                        "Subscriptions",
                        "Office - Tools & Consumables",
                        "Project - Consumables",
                        "Transportation",
                        "Project Expenses - Miscellaneous",
                        "Office Expenses - Miscellaneous",
                        "Can't classify",
                    ]
                    cat_index = (
                        category_options.index(default_cat)
                        if default_cat in category_options
                        else 7
                    )
                    category = st.selectbox(
                        "Category", category_options, index=cat_index
                    )

                    current_month = get_current_month_str(date_input)
                    previous_codes = get_project_codes_for_month(
                        month_str=current_month
                    )
                    code_options = ["Enter new project code..."] + previous_codes
                    selected_code_option = st.selectbox(
                        "Project Code", options=code_options, index=0
                    )

                    if selected_code_option == "Enter new project code...":
                        project_code = st.text_input(
                            "Enter New Project Code", placeholder="e.g., 250909-PDS-303"
                        )
                    else:
                        project_code = selected_code_option

                with c2:
                    if category == "Food & Beverages":
                        meal = get_meal_description(
                            default_time if default_time else datetime.now()
                        )
                        description = st.text_input(
                            "Description", value=default_desc or meal
                        )
                    else:
                        description = st.text_input("Description", value=default_desc)

                    amount = st.number_input(
                        "Amount (SAR/AED)",
                        min_value=0.0,
                        step=0.5,
                        value=float(default_amt),
                    )

                    # Checkbox for capping food bills - only show for Food & Beverages
                    cap_to_40 = False
                    if category == "Food & Beverages":
                        cap_to_40 = st.checkbox(
                            "Cap amount to 40 SAR/AED",
                            value=False,
                            key="cap_food_bill_checkbox",
                            help="If checked, amount will be capped at 40 and noted in description",
                        )

                submitted = st.form_submit_button(
                    "✅ Save Receipt Expense", use_container_width=True
                )

                if submitted:
                    final_desc = description.strip()
                    final_amt = float(amount)

                    # Only cap if checkbox is checked, category is Food & Beverages, and amount > 40
                    if category == "Food & Beverages" and cap_to_40 and final_amt > 40:
                        final_desc = f"{final_desc} (capped at 40)"
                        final_amt = 40.0
                        st.warning("Amount capped at 40")

                    temp_filename = f"temp_{datetime.now().timestamp()}.jpg"
                    save_path = os.path.join(IMAGES_DIR, temp_filename)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    df = load_data_for(None)
                    new_row = {
                        "Ref": len(df) + 1,
                        "Date": formatted_date,
                        "Description": final_desc,
                        "Category": category,
                        "Project Code": project_code,
                        "Amount": final_amt,
                        "Original_Image_Path": temp_filename,
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data_current(df)
                    st.success("Saved!")
                    st.rerun()

    st.divider()
    st.subheader("📦 Generate Receipts Package")
    st.write(f"Selected month: **{month_view}**")
    if st.button("Generate Receipts ZIP", use_container_width=True):
        try:
            generate_receipts_package(report_month, selected_month_slug)
        except Exception as e:
            st.error("Failed to generate receipts package.")
            st.exception(e)

# =========================================================
# TAB 2: CREDIT CARD
# =========================================================
with tab2:
    st.subheader("Credit Card")

    if month_view != "CURRENT":
        st.warning(
            "You are viewing archived data (read-only). Switch to CURRENT to add new credit card expenses."
        )

    cc_view = load_cc_for(selected_month_slug)
    if not cc_view.empty:
        st.dataframe(
            cc_view[["Date", "Description", "Category", "Project Code", "Amount"]],
            use_container_width=True,
        )
        st.metric("Total Credit Card", f"{cc_view['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No credit card expenses for this month selection.")

    if month_view == "CURRENT":
        uploaded_statement = st.file_uploader(
            "Upload Credit Card Statement (Camera or File)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=False,
            help="📷 Tap to take a photo with your camera or browse files",
            key="cc_upload",
        )

        ai_data = {}
        if uploaded_statement:
            st.image(
                uploaded_statement,
                caption="Statement Preview",
                use_container_width=True,
            )
            h = file_hash(uploaded_statement)

            analyze_cc_btn = st.button(
                "🤖 Analyze Statement with AI",
                key="cc_analyze_btn",
                use_container_width=True,
            )
            clear_cc_btn = st.button(
                "♻️ Clear AI Result", key="cc_clear_btn", use_container_width=True
            )

            if clear_cc_btn:
                st.session_state.pop("cc_ai_hash", None)
                st.session_state.pop("cc_ai_data", None)

            if analyze_cc_btn:
                if st.session_state.get("cc_ai_hash") == h and st.session_state.get(
                    "cc_ai_data"
                ):
                    st.info("Using cached AI result.")
                else:
                    with st.spinner("Reading statement with Gemini..."):
                        data = analyze_credit_card_statement(uploaded_statement)
                        if data:
                            st.session_state["cc_ai_data"] = data
                            st.session_state["cc_ai_hash"] = h
                            st.toast("Statement AI completed!")

            ai_data = st.session_state.get("cc_ai_data", {})

        default_date = datetime.today()
        default_desc = ""
        default_cat = "Project - Consumables"
        default_amt = 0.0

        if ai_data:
            try:
                default_date = datetime.strptime(ai_data.get("date"), "%Y-%m-%d")
            except:
                pass
            default_desc = ai_data.get("description", "")
            default_cat = ai_data.get("category", default_cat)
            try:
                default_amt = float(ai_data.get("amount", 0.0))
            except:
                pass

        with st.form("cc_form"):
            c1, c2 = st.columns(2)
            with c1:
                date_input = st.date_input("Date", value=default_date)
                formatted_date = date_input.strftime("%d-%b")

                category_options = [
                    "Hotel Booking",
                    "Food & Beverages",
                    "Visa & Ticket",
                    "Parking",
                    "R & D Expenses",
                    "Subscriptions",
                    "Office - Tools & Consumables",
                    "Project - Consumables",
                    "Transportation",
                    "Project Expenses - Miscellaneous",
                    "Office Expenses - Miscellaneous",
                    "Can't classify",
                ]
                cat_index = (
                    category_options.index(default_cat)
                    if default_cat in category_options
                    else 7
                )
                category = st.selectbox("Category", category_options, index=cat_index)

                current_month = get_current_month_str(date_input)
                previous_codes = get_project_codes_for_month(month_str=current_month)
                code_options = ["Enter new project code..."] + previous_codes
                selected_code_option = st.selectbox(
                    "Project Code", options=code_options, index=0
                )

                if selected_code_option == "Enter new project code...":
                    project_code = st.text_input(
                        "Enter New Project Code", placeholder="e.g., 250909-PDS-303"
                    )
                else:
                    project_code = selected_code_option

            with c2:
                description = st.text_input("Description", value=default_desc)
                amount = st.number_input(
                    "Amount (SAR/AED)",
                    min_value=0.0,
                    step=0.5,
                    value=float(default_amt),
                )

            submitted = st.form_submit_button(
                "✅ Save Credit Card Expense", use_container_width=True
            )

            if submitted:
                if not description.strip():
                    st.error("Description required.")
                else:
                    cc_df = load_cc_for(None)
                    new_row = {
                        "Date": formatted_date,
                        "Description": description.strip(),
                        "Category": category,
                        "Project Code": project_code if project_code else "",
                        "Amount": float(amount),
                    }
                    cc_df = pd.concat(
                        [cc_df, pd.DataFrame([new_row])], ignore_index=True
                    )
                    save_cc_current(cc_df)
                    st.success("Saved!")
                    st.rerun()

    st.divider()
    st.subheader("📦 Generate Credit Card Package")
    st.write(f"Selected month: **{month_view}**")
    if st.button("Generate Credit Card ZIP", use_container_width=True):
        try:
            generate_credit_card_package(report_month, selected_month_slug)
        except Exception as e:
            st.error("Failed to generate credit card package.")
            st.exception(e)

# =========================================================
# TAB 3: TRANSPORTATION
# =========================================================
with tab3:
    st.subheader("🚗 Transportation Expenses")

    if month_view != "CURRENT":
        st.warning(
            f"⚠️ You are viewing archived month: **{month_view}**. "
            "Switch to CURRENT to add new transportation expenses."
        )

    # Form for Transportation Entry
    with st.form("transport_form"):
        col1, col2 = st.columns(2)

        with col1:
            travel_date = st.date_input("Date *", value=datetime.today())
            formatted_date = travel_date.strftime("%d-%b")

            from_location = st.text_input(
                "From *",
                placeholder="e.g., Dubai, UAE",
                help="Enter the origin location",
            )

            destination = st.text_input(
                "Destination *",
                placeholder="e.g., Abu Dhabi, UAE",
                help="Enter the destination location",
            )

        with col2:
            return_included = st.checkbox(
                "Return Included",
                help="Check if this trip includes return journey",
            )

            # Project Code (optional)
            current_month = get_current_month_str(travel_date)
            previous_codes = get_project_codes_for_month(month_str=current_month)
            code_options = ["Enter new project code..."] + previous_codes
            selected_code_option = st.selectbox(
                "Project Code (Optional)",
                options=code_options,
                index=0,
                help="Select a previously used code or enter a new one",
            )

            if selected_code_option == "Enter new project code...":
                project_code = st.text_input(
                    "Enter New Project Code",
                    placeholder="e.g., 250909-PDS-303",
                )
            else:
                project_code = selected_code_option

        submitted = st.form_submit_button(
            "✅ Save Transportation Expense", use_container_width=True
        )

        if submitted:
            # Validation
            errors = []
            if not from_location or not from_location.strip():
                errors.append("Please enter From location")
            if not destination or not destination.strip():
                errors.append("Please enter Destination")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save to CSV
                transport_df = load_transport_data(None)  # Load current month

                new_row = {
                    "Date": formatted_date,
                    "From": from_location.strip(),
                    "Destination": destination.strip(),
                    "Return Included": return_included,
                    "Project Code": project_code if project_code else "",
                }

                transport_df = pd.concat(
                    [transport_df, pd.DataFrame([new_row])], ignore_index=True
                )
                save_transport_data(transport_df)

                st.success("Transportation expense saved!")
                st.rerun()

    # Display Current Month Transport Records
    st.divider()
    st.subheader("Current Month Transportation Expenses")
    transport_df = load_transport_data(selected_month_slug)

    if not transport_df.empty:
        # Filter by report month if needed
        try:
            report_month_obj = datetime.strptime(report_month, "%B %Y")
            transport_df["Date_parsed"] = pd.to_datetime(
                transport_df["Date"], format="%d-%b", errors="coerce"
            )
            month_filter = (
                transport_df["Date_parsed"].dt.year == report_month_obj.year
            ) & (transport_df["Date_parsed"].dt.month == report_month_obj.month)
            month_transport = transport_df[month_filter].copy()
            month_transport = month_transport.drop(columns=["Date_parsed"])
        except:
            # Fallback: show all if parsing fails
            month_transport = transport_df.copy()

        if not month_transport.empty:
            # Prepare display dataframe
            display_df = month_transport[
                ["Date", "From", "Destination", "Return Included", "Project Code"]
            ].copy()

            # Convert boolean to readable text
            display_df["Return Included"] = display_df["Return Included"].apply(
                lambda x: "Yes" if x else "No"
            )

            # Handle empty/null project codes
            display_df["Project Code"] = (
                display_df["Project Code"].fillna("").astype(str)
            )
            display_df["Project Code"] = display_df["Project Code"].apply(
                lambda x: (
                    x[:-2] if x.endswith(".0") and x.replace(".0", "").isdigit() else x
                )
            )

            st.dataframe(display_df, use_container_width=True)

            # Generate Excel button
            st.divider()
            st.subheader("Generate Report")
            if st.button("📊 Generate Transportation Excel", use_container_width=True):
                excel_path = "Transportation_Expenses.xlsx"
                generate_transport_excel(month_transport, excel_path)
                st.success("Excel file generated!")
                with open(excel_path, "rb") as fp:
                    st.download_button(
                        "⬇️ Download Transportation Expenses Excel",
                        fp,
                        "Transportation_Expenses.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
        else:
            st.info(f"No transport records found for {report_month}.")
    else:
        st.info("No transport records yet.")
