import streamlit as st
import pandas as pd
import os
import json
import re
import time as time_module
import hashlib
from datetime import datetime, time as dt_time
from PIL import Image
import google.generativeai as genai

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

import io

# =========================================================
# APP INFO
# =========================================================
APP_VERSION = "v2.1.0 - Google Sheets + Drive Storage (No ZIP)"

# =========================================================
# SECRETS / CONFIG
# =========================================================
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

SHEET_NAME = st.secrets.get("SHEET_NAME", "PurpleGlo_Expenses")
DRIVE_FOLDER_ID = st.secrets.get("DRIVE_FOLDER_ID", "")

# google_credentials must be in Streamlit secrets
# [google_credentials] ... service account json fields ...
credentials_dict = dict(st.secrets["google_credentials"])

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
sheets_client = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

# =========================================================
# SHEET HEADERS
# =========================================================
RECEIPTS_HEADERS = [
    "Report Month",
    "Ref",
    "Date",
    "Description",
    "Category",
    "Project Code",
    "Project Name",
    "Amount",
    "DriveFileId",
    "DriveFileName",
    "DriveLink",
]

CREDIT_HEADERS = [
    "Report Month",
    "Ref",
    "Date",
    "Description",
    "Category",
    "Project Code",
    "Project Name",
    "Amount",
    "DriveFileId",
    "DriveFileName",
    "DriveLink",
]

TRANSPORT_HEADERS = [
    "Report Month",
    "Ref",
    "Date",
    "From",
    "Destination",
    "Return Included",
    "Project Code",
    "Project Name",
]

CATEGORY_OPTIONS = [
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


def get_current_month_str(date_obj=None):
    if date_obj is None:
        date_obj = datetime.now()
    elif isinstance(date_obj, datetime):
        date_obj = date_obj.date()
    return date_obj.strftime("%b")


def get_meal_description(time_obj):
    if isinstance(time_obj, datetime):
        time_obj = time_obj.time()
    elif not isinstance(time_obj, dt_time):
        if isinstance(time_obj, str):
            try:
                hour, minute = map(int, time_obj.split(":"))
                time_obj = dt_time(hour, minute)
            except Exception:
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
    if not text:
        raise ValueError("Empty response from Gemini.")

    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace : last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON object found in Gemini response. Response: {text[:200]}")


# =========================================================
# GOOGLE SHEETS HELPERS
# =========================================================
def _open_sheet():
    try:
        return sheets_client.open(SHEET_NAME)
    except Exception as e:
        st.error(
            f"Failed to open Google Sheet '{SHEET_NAME}'. "
            f"Make sure the Sheet is shared with the service account email.\n\nError: {e}"
        )
        raise


def get_worksheet(sheet_name: str, headers: list[str]):
    sheet = _open_sheet()
    try:
        ws = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title=sheet_name, rows="3000", cols="30")

    # Ensure header row is correct
    first_row = ws.row_values(1)
    if first_row != headers:
        ws.clear()
        ws.append_row(headers)

    return ws


def load_df(sheet_name: str, headers: list[str]) -> pd.DataFrame:
    try:
        ws = get_worksheet(sheet_name, headers)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=headers)
        df = pd.DataFrame(records)
        # Ensure all columns exist
        for c in headers:
            if c not in df.columns:
                df[c] = ""
        return df[headers].copy()
    except Exception as e:
        st.error(f"Failed to load '{sheet_name}' from Sheets: {e}")
        return pd.DataFrame(columns=headers)


def append_row(sheet_name: str, headers: list[str], row: dict) -> bool:
    try:
        ws = get_worksheet(sheet_name, headers)
        ws.append_row([row.get(h, "") for h in headers], value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        st.error(f"Failed to append row to '{sheet_name}': {e}")
        return False


def delete_row_by_ref_and_month(sheet_name: str, headers: list[str], report_month: str, ref: int):
    """
    Deletes the first matching row where Report Month == report_month and Ref == ref.
    Returns (success, message, drive_file_id)
    """
    try:
        ws = get_worksheet(sheet_name, headers)
        records = ws.get_all_records()

        target_row_index = None
        drive_file_id = ""

        for i, r in enumerate(records, start=2):  # header is row 1
            if str(r.get("Report Month", "")).strip() == str(report_month).strip() and str(r.get("Ref", "")).strip() == str(ref):
                target_row_index = i
                drive_file_id = str(r.get("DriveFileId", "")).strip()
                break

        if not target_row_index:
            return False, f"Row not found: {report_month} Ref {ref}", ""

        ws.delete_rows(target_row_index)
        return True, f"Deleted: {report_month} Ref {ref}", drive_file_id

    except Exception as e:
        return False, f"Error deleting row: {e}", ""


# =========================================================
# GOOGLE DRIVE HELPERS
# =========================================================
def upload_image_to_drive(uploaded_file, report_month: str, ref: int) -> tuple[str, str, str]:
    """
    Upload image bytes to Drive folder.
    Filename starts with Ref (sl number), per user requirement.
    Returns (file_id, file_name, webViewLink)
    """
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("DRIVE_FOLDER_ID missing in Streamlit secrets.")

    file_bytes = uploaded_file.getvalue()
    month_slug = slugify(report_month)

    # Filename begins with Ref. Also includes month to avoid confusion.
    filename = f"{ref}_{month_slug}.jpg"

    media = MediaIoBaseUpload(
        io.BytesIO(file_bytes),
        mimetype=uploaded_file.type or "image/jpeg",
        resumable=False,
    )

    meta = {"name": filename, "parents": [DRIVE_FOLDER_ID]}

    created = drive_service.files().create(
        body=meta,
        media_body=media,
        fields="id,name,webViewLink",
    ).execute()

    return created["id"], created["name"], created.get("webViewLink", "")


def delete_drive_file(file_id: str):
    if not file_id:
        return
    try:
        drive_service.files().delete(fileId=file_id).execute()
    except Exception:
        # do not block delete if drive deletion fails
        pass


# =========================================================
# GEMINI AI
# =========================================================
def analyze_receipt_gemini(image_file):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not set.")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_file)

    prompt = """Analyze this receipt image and extract the following information.

Return ONLY a valid JSON object with these exact keys:
- "date": YYYY-MM-DD format (required)
- "time": HH:MM format (24-hour) or null if not available
- "amount": numeric value only (required)
- "description": short description text (required)
- "category": must be one of these exact strings:
  "Hotel Booking", "Food & Beverages", "Visa & Ticket", "Parking", "R & D Expenses", "Subscriptions",
  "Office - Tools & Consumables", "Project - Consumables", "Transportation",
  "Project Expenses - Miscellaneous", "Office Expenses - Miscellaneous", "Can't classify"

Example output format:
{"date": "2024-01-15", "time": "14:30", "amount": 45.50, "description": "Restaurant meal", "category": "Food & Beverages"}

Return ONLY the JSON object, no other text or explanation."""

    generation_config = {"temperature": 0, "max_output_tokens": 1000}

    try:
        response = model.generate_content([prompt, img], generation_config=generation_config)
        if not response or not response.text:
            st.error("AI Error: Empty response from Gemini API.")
            return None
        return _extract_json(response.text)
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            wait_s = parse_retry_seconds(msg) or 30
            st.warning(f"Rate limit hit. Waiting {wait_s}s then retrying once...")
            time_module.sleep(wait_s)
            try:
                response = model.generate_content([prompt, img], generation_config=generation_config)
                if not response or not response.text:
                    st.error("AI Error: Empty response from Gemini API on retry.")
                    return None
                return _extract_json(response.text)
            except Exception as e2:
                st.error(f"AI Error on retry: {e2}")
                return None
        st.error(f"AI Error: {e}")
        return None


def analyze_credit_card_statement(image_file):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not set.")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_file)

    prompt = """Analyze this credit card statement image. Extract TOTAL expense as JSON:
{"date":"YYYY-MM-DD","description":"...","amount":123.45,"category":"..."}
Category must be one of:
["Hotel Booking","Food & Beverages","Visa & Ticket","Parking","R & D Expenses","Subscriptions",
"Office - Tools & Consumables","Project - Consumables","Transportation",
"Project Expenses - Miscellaneous","Office Expenses - Miscellaneous","Can't classify"]

Return ONLY JSON."""

    generation_config = {"temperature": 0, "max_output_tokens": 1000}

    try:
        response = model.generate_content([prompt, img], generation_config=generation_config)
        if not response or not response.text:
            return None
        return _extract_json(response.text)
    except Exception as e:
        msg = str(e)
        if "429" in msg:
            wait_s = parse_retry_seconds(msg) or 30
            st.warning(f"Rate limit hit. Waiting {wait_s}s then retrying once...")
            time_module.sleep(wait_s)
            response = model.generate_content([prompt, img], generation_config=generation_config)
            return _extract_json(response.text)
        st.error(f"AI Error: {e}")
        return None


# =========================================================
# BUSINESS LOGIC
# =========================================================
def get_next_ref_for_month(df: pd.DataFrame, report_month: str) -> int:
    if df.empty:
        return 1
    m = df[df["Report Month"].astype(str).str.strip() == str(report_month).strip()]
    if m.empty:
        return 1
    refs = pd.to_numeric(m["Ref"], errors="coerce").fillna(0).astype(int)
    return int(refs.max()) + 1


def get_project_codes_for_month(report_month: str, include_credit_card=True):
    codes = []

    r_df = load_df("Receipts", RECEIPTS_HEADERS)
    if not r_df.empty:
        m = r_df[r_df["Report Month"].astype(str).str.strip() == str(report_month).strip()]
        for c in m["Project Code"].dropna():
            s = str(c).strip()
            if s:
                codes.append(s)

    if include_credit_card:
        cc_df = load_df("CreditCard", CREDIT_HEADERS)
        if not cc_df.empty:
            m2 = cc_df[cc_df["Report Month"].astype(str).str.strip() == str(report_month).strip()]
            for c in m2["Project Code"].dropna():
                s = str(c).strip()
                if s:
                    codes.append(s)

    return sorted(list(set(codes)))


# =========================================================
# UI
# =========================================================
st.set_page_config(
    page_title="PurpleGlo Expense Manager",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.markdown(
    """
    <style>
    .stButton > button { min-height: 48px; font-size: 16px; }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select { font-size: 16px; padding: 12px; }
    .stDateInput > div > div > input { font-size: 16px; padding: 12px; }
    .dataframe { overflow-x: auto; display: block; }
    .version-badge {
        position: fixed; bottom: 10px; right: 10px;
        background: rgba(120, 120, 255, 0.85);
        color: white; padding: 6px 10px; border-radius: 6px;
        font-size: 12px; z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🟣 PurpleGlo Expense Manager")
st.markdown(f'<div class="version-badge">{APP_VERSION}</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Settings")
report_month = st.sidebar.text_input("Report Month", value=datetime.now().strftime("%b %Y"))
st.sidebar.write("Storage: Google Sheets + Google Drive ✅")

if not DRIVE_FOLDER_ID:
    st.sidebar.error("Missing DRIVE_FOLDER_ID in secrets.")

tab1, tab2, tab3 = st.tabs(["📸 Receipts", "💳 Credit Card", "🚗 Transportation"])


# =========================================================
# TAB 1: RECEIPTS
# =========================================================
with tab1:
    st.subheader(f"Receipts ({report_month})")

    df_all = load_df("Receipts", RECEIPTS_HEADERS)
    df_view = df_all[df_all["Report Month"].astype(str).str.strip() == str(report_month).strip()].copy()

    if not df_view.empty:
        show_cols = ["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount"]
        display_df = df_view[show_cols].copy()
        display_df["Amount"] = pd.to_numeric(display_df["Amount"], errors="coerce").fillna(0.0)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.metric("Total Receipts", f"{display_df['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No receipt expenses for this month yet.")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Receipt (Camera or File)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        help="Take photo or upload receipt image",
        key="receipt_upload",
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Receipt Preview", use_container_width=True)
        h = file_hash(uploaded_file)

        # Auto analyze with Gemini (cached per image)
        if st.session_state.get("receipt_ai_hash") == h and st.session_state.get("receipt_ai_data"):
            ai_data = st.session_state.get("receipt_ai_data", {})
        else:
            with st.spinner("Reading receipt with Gemini..."):
                ai_data = analyze_receipt_gemini(uploaded_file)
                if ai_data:
                    st.session_state["receipt_ai_data"] = ai_data
                    st.session_state["receipt_ai_hash"] = h
                    st.toast("Receipt analyzed successfully!")
                else:
                    ai_data = {}

        default_date = datetime.today()
        default_desc = ""
        default_cat = "Project - Consumables"
        default_amt = 0.0
        default_time = None

        if ai_data:
            try:
                default_date = datetime.strptime(ai_data.get("date"), "%Y-%m-%d")
            except Exception:
                pass
            default_time = ai_data.get("time")
            default_cat = ai_data.get("category", default_cat)
            try:
                default_amt = float(ai_data.get("amount", default_amt))
            except Exception:
                pass

            if default_cat == "Food & Beverages":
                default_desc = get_meal_description(default_time if default_time else datetime.now())
            else:
                default_desc = ai_data.get("description", "")

        with st.form("receipt_form"):
            c1, c2 = st.columns([1, 1])

            with c1:
                date_input = st.date_input("Date", value=default_date)
                formatted_date = date_input.strftime("%d-%b")

                cat_index = CATEGORY_OPTIONS.index(default_cat) if default_cat in CATEGORY_OPTIONS else 7
                category = st.selectbox("Category", CATEGORY_OPTIONS, index=cat_index)

                previous_codes = get_project_codes_for_month(report_month, include_credit_card=True)
                code_options = ["Enter new project code..."] + previous_codes
                selected_code_option = st.selectbox("Project Code", options=code_options, index=0)

                if selected_code_option == "Enter new project code...":
                    project_code = st.text_input("Enter New Project Code", placeholder="e.g., 250909-PDS-303")
                else:
                    project_code = selected_code_option

                project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha")

            with c2:
                if category == "Food & Beverages":
                    meal = get_meal_description(default_time if default_time else datetime.now())
                    description = st.text_input("Description", value=default_desc or meal)
                else:
                    description = st.text_input("Description", value=default_desc)

                amount = st.number_input("Amount (SAR/AED)", min_value=0.0, step=0.5, value=float(default_amt))

                cap_to_40 = False
                if category == "Food & Beverages":
                    cap_to_40 = st.checkbox("Cap amount to 40 SAR/AED", value=False, key="cap_food_bill_checkbox")

            submitted = st.form_submit_button("✅ Save Receipt Expense", use_container_width=True)

            if submitted:
                final_project_name = (project_name or "").strip()
                if not final_project_name:
                    st.error("Project Name is required.")
                else:
                    final_desc = (description or "").strip()
                    final_amt = float(amount)

                    if category == "Food & Beverages" and cap_to_40 and final_amt > 40:
                        final_desc = f"{final_desc} (capped at 40)"
                        final_amt = 40.0

                    # Next Ref per month
                    new_ref = get_next_ref_for_month(df_all, report_month)

                    # Upload image to Drive
                    try:
                        with st.spinner("Uploading image to Google Drive..."):
                            drive_id, drive_name, drive_link = upload_image_to_drive(uploaded_file, report_month, new_ref)
                    except Exception as e:
                        st.error(f"Drive upload failed: {e}")
                        st.stop()

                    row = {
                        "Report Month": report_month,
                        "Ref": new_ref,
                        "Date": formatted_date,
                        "Description": final_desc,
                        "Category": category,
                        "Project Code": project_code,
                        "Project Name": final_project_name,
                        "Amount": final_amt,
                        "DriveFileId": drive_id,
                        "DriveFileName": drive_name,
                        "DriveLink": drive_link,
                    }

                    with st.spinner("Saving to Google Sheets..."):
                        ok = append_row("Receipts", RECEIPTS_HEADERS, row)

                    if ok:
                        st.success("Saved!")
                        # Clear cached image analysis and uploader
                        for k in ["receipt_upload", "receipt_ai_hash", "receipt_ai_data", "cap_food_bill_checkbox"]:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()

    st.divider()
    st.subheader("🗑️ Delete Receipts")
    if not df_view.empty:
        # Create buttons by Ref
        refs = sorted(df_view["Ref"].astype(int).tolist())
        cols = st.columns(3)
        for i, ref in enumerate(refs):
            with cols[i % 3]:
                if st.button(f"Delete #{ref}", key=f"del_receipt_{ref}", use_container_width=True):
                    success, msg, drive_id = delete_row_by_ref_and_month("Receipts", RECEIPTS_HEADERS, report_month, int(ref))
                    if success:
                        delete_drive_file(drive_id)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Nothing to delete for this month.")


# =========================================================
# TAB 2: CREDIT CARD
# =========================================================
with tab2:
    st.subheader(f"Credit Card ({report_month})")

    cc_all = load_df("CreditCard", CREDIT_HEADERS)
    cc_view = cc_all[cc_all["Report Month"].astype(str).str.strip() == str(report_month).strip()].copy()

    if not cc_view.empty:
        show_cols = ["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount"]
        display_df = cc_view[show_cols].copy()
        display_df["Amount"] = pd.to_numeric(display_df["Amount"], errors="coerce").fillna(0.0)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.metric("Total Credit Card", f"{display_df['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No credit card expenses for this month yet.")

    st.divider()

    uploaded_statement = st.file_uploader(
        "Upload Credit Card Statement (Image)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        help="Upload statement image (will be saved to Drive)",
        key="cc_upload",
    )

    ai_data = {}
    if uploaded_statement:
        st.image(uploaded_statement, caption="Statement Preview", use_container_width=True)
        h = file_hash(uploaded_statement)

        if st.session_state.get("cc_ai_hash") == h and st.session_state.get("cc_ai_data"):
            ai_data = st.session_state.get("cc_ai_data", {})
        else:
            with st.spinner("Reading statement with Gemini..."):
                data = analyze_credit_card_statement(uploaded_statement)
                if data:
                    st.session_state["cc_ai_data"] = data
                    st.session_state["cc_ai_hash"] = h
                    st.toast("Statement analyzed successfully!")
                    ai_data = data
                else:
                    ai_data = {}

    default_date = datetime.today()
    default_desc = ""
    default_cat = "Project - Consumables"
    default_amt = 0.0

    if ai_data:
        try:
            default_date = datetime.strptime(ai_data.get("date"), "%Y-%m-%d")
        except Exception:
            pass
        default_desc = ai_data.get("description", "")
        default_cat = ai_data.get("category", default_cat)
        try:
            default_amt = float(ai_data.get("amount", 0.0))
        except Exception:
            pass

    with st.form("cc_form"):
        c1, c2 = st.columns([1, 1])
        with c1:
            date_input = st.date_input("Date", value=default_date)
            formatted_date = date_input.strftime("%d-%b")

            cat_index = CATEGORY_OPTIONS.index(default_cat) if default_cat in CATEGORY_OPTIONS else 7
            category = st.selectbox("Category", CATEGORY_OPTIONS, index=cat_index, key="cc_cat")

            previous_codes = get_project_codes_for_month(report_month, include_credit_card=True)
            code_options = ["Enter new project code..."] + previous_codes
            selected_code_option = st.selectbox("Project Code", options=code_options, index=0, key="cc_code_sel")

            if selected_code_option == "Enter new project code...":
                project_code = st.text_input("Enter New Project Code", placeholder="e.g., 250909-PDS-303", key="cc_code_new")
            else:
                project_code = selected_code_option

            project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha", key="cc_projname")

        with c2:
            description = st.text_input("Description", value=default_desc, key="cc_desc")
            amount = st.number_input("Amount (SAR/AED)", min_value=0.0, step=0.5, value=float(default_amt), key="cc_amt")

            cap_to_40 = False
            if category == "Food & Beverages":
                cap_to_40 = st.checkbox("Cap amount to 40 SAR/AED", value=False, key="cc_cap_food_bill_checkbox")

        submitted = st.form_submit_button("✅ Save Credit Card Expense", use_container_width=True)

        if submitted:
            if not (project_name or "").strip():
                st.error("Project Name is required.")
            elif not (description or "").strip():
                st.error("Description required.")
            else:
                final_desc = description.strip()
                final_amt = float(amount)
                final_project_name = project_name.strip()

                if category == "Food & Beverages" and cap_to_40 and final_amt > 40:
                    final_desc = f"{final_desc} (capped at 40)"
                    final_amt = 40.0

                new_ref = get_next_ref_for_month(cc_all, report_month)

                # Statement image upload optional (but recommended)
                drive_id = ""
                drive_name = ""
                drive_link = ""
                if uploaded_statement:
                    try:
                        with st.spinner("Uploading statement to Google Drive..."):
                            drive_id, drive_name, drive_link = upload_image_to_drive(uploaded_statement, report_month, new_ref)
                    except Exception as e:
                        st.error(f"Drive upload failed: {e}")
                        st.stop()

                row = {
                    "Report Month": report_month,
                    "Ref": new_ref,
                    "Date": formatted_date,
                    "Description": final_desc,
                    "Category": category,
                    "Project Code": project_code if project_code else "",
                    "Project Name": final_project_name,
                    "Amount": final_amt,
                    "DriveFileId": drive_id,
                    "DriveFileName": drive_name,
                    "DriveLink": drive_link,
                }

                with st.spinner("Saving to Google Sheets..."):
                    ok = append_row("CreditCard", CREDIT_HEADERS, row)

                if ok:
                    st.success("Saved!")
                    for k in ["cc_upload", "cc_ai_hash", "cc_ai_data", "cc_cap_food_bill_checkbox"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

    st.divider()
    st.subheader("🗑️ Delete Credit Card Rows")
    if not cc_view.empty:
        refs = sorted(cc_view["Ref"].astype(int).tolist())
        cols = st.columns(3)
        for i, ref in enumerate(refs):
            with cols[i % 3]:
                if st.button(f"Delete #{ref}", key=f"del_cc_{ref}", use_container_width=True):
                    success, msg, drive_id = delete_row_by_ref_and_month("CreditCard", CREDIT_HEADERS, report_month, int(ref))
                    if success:
                        delete_drive_file(drive_id)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Nothing to delete for this month.")


# =========================================================
# TAB 3: TRANSPORTATION
# =========================================================
with tab3:
    st.subheader(f"Transportation ({report_month})")

    t_all = load_df("Transport", TRANSPORT_HEADERS)
    t_view = t_all[t_all["Report Month"].astype(str).str.strip() == str(report_month).strip()].copy()

    if not t_view.empty:
        show_cols = ["Ref", "Date", "From", "Destination", "Return Included", "Project Code", "Project Name"]
        display_df = t_view[show_cols].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transport expenses for this month yet.")

    st.divider()

    with st.form("transport_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            travel_date = st.date_input("Date *", value=datetime.today())
            formatted_date = travel_date.strftime("%d-%b")
            from_location = st.text_input("From *", placeholder="e.g., Dubai, UAE")
            destination = st.text_input("Destination *", placeholder="e.g., Abu Dhabi, UAE")

        with col2:
            return_included = st.checkbox("Return Included")

            previous_codes = get_project_codes_for_month(report_month, include_credit_card=True)
            code_options = ["Enter new project code..."] + previous_codes
            selected_code_option = st.selectbox("Project Code (Optional)", options=code_options, index=0, key="tr_code_sel")

            if selected_code_option == "Enter new project code...":
                project_code = st.text_input("Enter New Project Code", placeholder="e.g., 250909-PDS-303", key="tr_code_new")
            else:
                project_code = selected_code_option

            project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha", key="tr_projname")

        submitted = st.form_submit_button("✅ Save Transportation Expense", use_container_width=True)

        if submitted:
            errors = []
            if not (from_location or "").strip():
                errors.append("Please enter From location")
            if not (destination or "").strip():
                errors.append("Please enter Destination")
            if not (project_name or "").strip():
                errors.append("Please enter Project Name")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                new_ref = get_next_ref_for_month(t_all, report_month)

                row = {
                    "Report Month": report_month,
                    "Ref": new_ref,
                    "Date": formatted_date,
                    "From": from_location.strip(),
                    "Destination": destination.strip(),
                    "Return Included": "Yes" if return_included else "No",
                    "Project Code": project_code if project_code else "",
                    "Project Name": project_name.strip(),
                }

                with st.spinner("Saving to Google Sheets..."):
                    ok = append_row("Transport", TRANSPORT_HEADERS, row)

                if ok:
                    st.success("Transportation saved!")
                    st.rerun()

    st.divider()
    st.subheader("🗑️ Delete Transport Rows")
    if not t_view.empty:
        refs = sorted(t_view["Ref"].astype(int).tolist())
        cols = st.columns(3)
        for i, ref in enumerate(refs):
            with cols[i % 3]:
                if st.button(f"Delete #{ref}", key=f"del_tr_{ref}", use_container_width=True):
                    # Transport has no drive file
                    success, msg, _ = delete_row_by_ref_and_month("Transport", TRANSPORT_HEADERS, report_month, int(ref))
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Nothing to delete for this month.")
