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
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import io
import tempfile
import shutil

# =========================================================
# GOOGLE SHEETS & DRIVE CONFIGURATION
# =========================================================

# Get credentials from Streamlit secrets
try:
    credentials_dict = dict(st.secrets["google_credentials"])
except:
    # Fallback to local file for testing
    credentials_dict = json.load(open("credentials.json"))

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, SCOPE)
sheets_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)

# Google Sheet and Drive settings
SHEET_NAME = "PurpleGlo_Expenses"  # Your Google Sheet name
DRIVE_FOLDER_ID = "1BG0cVmy77uWbJJ2ul3QJw0pYfkA95p0I"  # Your Drive folder ID

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

MAX_HISTORY_MONTHS = 3

# =========================================================
# GOOGLE SHEETS & DRIVE HELPERS
# =========================================================

def get_worksheet(sheet_name):
    """Get or create worksheet"""
    try:
        sheet = sheets_client.open(SHEET_NAME)
        return sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        sheet = sheets_client.open(SHEET_NAME)
        return sheet.add_worksheet(title=sheet_name, rows="1000", cols="20")
    except Exception as e:
        st.error(f"Failed to access Google Sheet '{SHEET_NAME}'. Make sure it's shared with the service account.")
        raise e

def upload_image_to_drive(image_file, filename):
    """Upload image to Google Drive and return shareable link and file ID"""
    try:
        # Reset file pointer
        image_file.seek(0)
        
        # Create file metadata
        file_metadata = {
            'name': filename,
            'parents': [DRIVE_FOLDER_ID]
        }
        
        # Create media content from BytesIO - use MediaIoBaseUpload for in-memory files
        media = MediaIoBaseUpload(
            io.BytesIO(image_file.getvalue()),
            mimetype='image/jpeg',
            resumable=True
        )
        
        # Upload to Drive
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, webContentLink'
        ).execute()
        
        # Make file publicly viewable
        drive_service.permissions().create(
            fileId=file['id'],
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        return file.get('webViewLink'), file.get('id')
        
    except Exception as e:
        st.error(f"Failed to upload image to Drive: {e}")
        return None, None

def delete_image_from_drive(image_id):
    """Delete image from Google Drive"""
    try:
        if image_id and image_id.strip():
            drive_service.files().delete(fileId=image_id).execute()
        return True
    except Exception as e:
        # Don't fail if image doesn't exist or already deleted
        return True

def download_image_from_drive(image_id):
    """Download image from Google Drive for export"""
    try:
        if not image_id or not image_id.strip():
            return None
        request = drive_service.files().get_media(fileId=image_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    except Exception as e:
        return None

def load_data_from_sheets(sheet_name):
    """Load data from Google Sheets"""
    try:
        worksheet = get_worksheet(sheet_name)
        data = worksheet.get_all_records()
        if not data:
            # Return empty DataFrame with correct columns
            if sheet_name == "Receipts":
                return pd.DataFrame(columns=[
                    "Ref", "Date", "Description", "Category", 
                    "Project Code", "Project Name", "Amount", 
                    "Image_Link", "Image_ID"
                ])
            elif sheet_name == "CreditCard":
                return pd.DataFrame(columns=[
                    "Date", "Description", "Category",
                    "Project Code", "Project Name", "Amount"
                ])
            elif sheet_name == "Transport":
                return pd.DataFrame(columns=[
                    "Date", "From", "Destination", "Return Included",
                    "Project Code", "Project Name"
                ])
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to load data from sheet '{sheet_name}': {e}")
        # Return empty DataFrame
        if sheet_name == "Receipts":
            return pd.DataFrame(columns=[
                "Ref", "Date", "Description", "Category", 
                "Project Code", "Project Name", "Amount", 
                "Image_Link", "Image_ID"
            ])
        elif sheet_name == "CreditCard":
            return pd.DataFrame(columns=[
                "Date", "Description", "Category",
                "Project Code", "Project Name", "Amount"
            ])
        elif sheet_name == "Transport":
            return pd.DataFrame(columns=[
                "Date", "From", "Destination", "Return Included",
                "Project Code", "Project Name"
            ])

def save_data_to_sheets(df, sheet_name):
    """Save DataFrame to Google Sheets"""
    try:
        worksheet = get_worksheet(sheet_name)
        # Clear existing data
        worksheet.clear()
        # Convert DataFrame to list of lists
        data = [df.columns.values.tolist()] + df.values.tolist()
        # Update the sheet
        worksheet.update(data, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Failed to save data to sheet '{sheet_name}': {e}")
        return False

# =========================================================
# DATA LOADING FUNCTIONS (Modified for Sheets)
# =========================================================

def load_data_for(selected_month_slug: str | None):
    """Load receipts data"""
    return load_data_from_sheets("Receipts")

def save_data_current(df: pd.DataFrame):
    """Save receipts data"""
    return save_data_to_sheets(df, "Receipts")

def load_cc_for(selected_month_slug: str | None):
    """Load credit card data"""
    return load_data_from_sheets("CreditCard")

def save_cc_current(df: pd.DataFrame):
    """Save credit card data"""
    return save_data_to_sheets(df, "CreditCard")

def load_transport_data(selected_month_slug: str | None = None):
    """Load transport data"""
    return load_data_from_sheets("Transport")

def save_transport_data(df: pd.DataFrame):
    """Save transport data"""
    return save_data_to_sheets(df, "Transport")

def delete_receipt(ref: int) -> tuple[bool, str]:
    """Delete receipt from Sheets and Drive"""
    try:
        df = load_data_for(None)
        
        row_to_delete = df[df["Ref"] == ref]
        if row_to_delete.empty:
            return False, f"Receipt {ref} not found."
        
        # Get image ID and delete from Drive
        if "Image_ID" in row_to_delete.columns:
            image_id = row_to_delete["Image_ID"].iloc[0]
            if pd.notna(image_id) and str(image_id).strip():
                delete_image_from_drive(str(image_id))
        
        # Remove from DataFrame
        df = df[df["Ref"] != ref]
        
        # Save to Sheets
        save_data_to_sheets(df, "Receipts")
        
        return True, f"Receipt {ref} deleted successfully."
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# =========================================================
# HELPER FUNCTIONS
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

def get_project_codes_for_month(month_str=None, date_obj=None, include_credit_card=True):
    if month_str is None:
        month_str = get_current_month_str(date_obj)

    all_codes = []
    df = load_data_for(None)
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
            cc_month_filter = cc_df["Date"].str.contains(f"-{month_str}$", case=False, na=False)
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
# GEMINI AI FUNCTIONS
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

    generation_config = {
        "temperature": 0,
        "max_output_tokens": 1000,
    }

    try:
        response = model.generate_content([prompt, img], generation_config=generation_config)
        if not response or not response.text:
            st.error("AI Error: Empty response from Gemini API.")
            return None
        return _extract_json(response.text)
    except ValueError as e:
        error_msg = str(e)
        st.error(f"AI Error: Failed to parse JSON from Gemini response. {error_msg}")
        if "Response:" in error_msg:
            st.code(error_msg.split("Response:")[1][:500], language="text")
        return None
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
            except ValueError as ve:
                st.error(f"AI Error: Failed to parse JSON on retry. {str(ve)}")
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
# EXPORT FUNCTIONS
# =========================================================

def generate_transport_excel(df: pd.DataFrame, output_path: str):
    """Generate Excel file for transport expenses"""
    if df.empty:
        st.warning("No transport expenses to export.")
        return

    excel_df = pd.DataFrame()
    excel_df["Date"] = df["Date"]
    excel_df["From"] = df["From"]
    excel_df["Destination"] = df["Destination"]
    excel_df["Return Included"] = df["Return Included"].apply(
        lambda x: "Return Included" if x else ""
    )
    excel_df["Project Code"] = df["Project Code"]
    if "Project Name" in df.columns:
        excel_df["Project Name"] = df["Project Name"].fillna("")

    excel_df.to_excel(output_path, index=False, engine="openpyxl")

def generate_receipts_package(report_month: str):
    """Generate receipts package with images downloaded from Drive"""
    df = load_data_for(None)
    
    if df.empty:
        st.error("No receipts expenses found.")
        return

    month_slug = slugify(report_month)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_folder = os.path.join(temp_dir, f"receipts_output_{month_slug}")
        bills_folder = os.path.join(output_folder, f"bills_{month_slug}")
        os.makedirs(bills_folder, exist_ok=True)

        # Download images from Drive
        if "Image_ID" in df.columns:
            progress_bar = st.progress(0)
            total = len(df)
            for idx, (_, row) in enumerate(df.iterrows()):
                image_id = row.get("Image_ID")
                if pd.notna(image_id) and str(image_id).strip():
                    image_data = download_image_from_drive(str(image_id))
                    if image_data:
                        dst = os.path.join(bills_folder, f"{row['Ref']}.jpg")
                        with open(dst, 'wb') as f:
                            f.write(image_data.read())
                progress_bar.progress((idx + 1) / total)
            progress_bar.empty()

        # Create Excel
        excel_path = os.path.join(output_folder, f"Receipts_Expenses_{month_slug}.xlsx")
        export_df = df.copy()
        if "Image_Link" in export_df.columns:
            export_df = export_df.drop(columns=["Image_Link"])
        if "Image_ID" in export_df.columns:
            export_df = export_df.drop(columns=["Image_ID"])
        export_df.to_excel(excel_path, index=False, engine="openpyxl")

        # Create ZIP
        zip_path = os.path.join(temp_dir, f"Receipts_Submission_{month_slug}")
        shutil.make_archive(zip_path, "zip", output_folder)

        st.success("Receipts package generated!")
        with open(f"{zip_path}.zip", "rb") as fp:
            st.download_button(
                "⬇️ Download Receipts ZIP",
                fp,
                file_name=f"Receipts_Submission_{month_slug}.zip",
                mime="application/zip",
            )

def generate_credit_card_package(report_month: str):
    """Generate credit card package"""
    cc_df = load_cc_for(None)
    if cc_df.empty:
        st.error("No credit card expenses found.")
        return

    month_slug = slugify(report_month)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_folder = os.path.join(temp_dir, f"credit_output_{month_slug}")
        os.makedirs(output_folder, exist_ok=True)

        excel_path = os.path.join(output_folder, f"CreditCard_Expenses_{month_slug}.xlsx")
        cc_df.to_excel(excel_path, index=False, engine="openpyxl")

        zip_path = os.path.join(temp_dir, f"CreditCard_Submission_{month_slug}")
        shutil.make_archive(zip_path, "zip", output_folder)

        st.success("Credit card package generated!")
        with open(f"{zip_path}.zip", "rb") as fp:
            st.download_button(
                "⬇️ Download Credit Card ZIP",
                fp,
                file_name=f"CreditCard_Submission_{month_slug}.zip",
                mime="application/zip",
            )

# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="PurpleGlo Expense Manager",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# Mobile-friendly CSS
st.markdown("""
    <style>
    .stButton > button {
        min-height: 48px;
        font-size: 16px;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        font-size: 16px;
        padding: 12px;
    }
    .element-container {
        margin-bottom: 1rem;
    }
    .stDateInput > div > div > input {
        font-size: 16px;
        padding: 12px;
    }
    .dataframe {
        overflow-x: auto;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🟣 PurpleGlo Expense Manager")

# Sidebar
st.sidebar.header("Settings")
report_month = st.sidebar.text_input("Report Month", value=datetime.now().strftime("%b %Y"))
st.sidebar.write("✅ Connected to Google Sheets")
st.sidebar.write("✅ Connected to Google Drive")

tab1, tab2, tab3 = st.tabs(["📸 Add Receipt", "💳 Credit Card", "🚗 Transportation"])

# =========================================================
# TAB 1: RECEIPTS
# =========================================================
with tab1:
    st.subheader("Receipts")

    df_view = load_data_for(None)
    if not df_view.empty:
        display_columns = ["Ref", "Date", "Description", "Category", "Project Code"]
        if "Project Name" in df_view.columns:
            display_columns.append("Project Name")
        display_columns.append("Amount")

        display_df = df_view[display_columns].copy()
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Delete buttons
        st.divider()
        st.write("**🗑️ Delete Receipts:**")
        num_cols = min(3, len(df_view)) if len(df_view) > 3 else max(2, len(df_view))
        delete_cols = st.columns(num_cols)

        for idx, row in df_view.iterrows():
            col_idx = idx % num_cols
            ref_value = int(row["Ref"])
            with delete_cols[col_idx]:
                if st.button(f"🗑️ Delete #{ref_value}", key=f"delete_receipt_{ref_value}", 
                            use_container_width=True):
                    success, message = delete_receipt(ref_value)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        st.metric("Total Receipts", f"{df_view['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No receipt expenses yet.")

    # Upload form
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

        # Auto-analyze
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
            except:
                pass
            default_time = ai_data.get("time")
            default_cat = ai_data.get("category", default_cat)
            try:
                default_amt = float(ai_data.get("amount", default_amt))
            except:
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

                category_options = [
                    "Hotel Booking", "Food & Beverages", "Visa & Ticket", "Parking",
                    "R & D Expenses", "Subscriptions", "Office - Tools & Consumables",
                    "Project - Consumables", "Transportation",
                    "Project Expenses - Miscellaneous", "Office Expenses - Miscellaneous",
                    "Can't classify",
                ]
                cat_index = category_options.index(default_cat) if default_cat in category_options else 7
                category = st.selectbox("Category", category_options, index=cat_index)

                current_month = get_current_month_str(date_input)
                previous_codes = get_project_codes_for_month(month_str=current_month)
                code_options = ["Enter new project code..."] + previous_codes
                selected_code_option = st.selectbox("Project Code", options=code_options, index=0)

                if selected_code_option == "Enter new project code...":
                    project_code = st.text_input("Enter New Project Code", 
                                                placeholder="e.g., 250909-PDS-303")
                else:
                    project_code = selected_code_option

                project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha",
                                            help="Enter the project name (required)")

            with c2:
                if category == "Food & Beverages":
                    meal = get_meal_description(default_time if default_time else datetime.now())
                    description = st.text_input("Description", value=default_desc or meal)
                else:
                    description = st.text_input("Description", value=default_desc)

                amount = st.number_input("Amount (SAR/AED)", min_value=0.0, step=0.5, 
                                        value=float(default_amt))

                cap_to_40 = False
                if category == "Food & Beverages":
                    cap_to_40 = st.checkbox("Cap amount to 40 SAR/AED", value=False,
                                           key="cap_food_bill_checkbox",
                                           help="If checked, amount will be capped at 40")

            submitted = st.form_submit_button("✅ Save Receipt Expense", use_container_width=True)

            if submitted:
                final_desc = description.strip()
                final_amt = float(amount)
                final_project_name = project_name.strip() if project_name else ""

                if not final_project_name:
                    st.error("Project Name is required.")
                else:
                    if category == "Food & Beverages" and cap_to_40 and final_amt > 40:
                        final_desc = f"{final_desc} (capped at 40)"
                        final_amt = 40.0
                        st.warning("Amount capped at 40")

                    # Upload image to Drive
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    df = load_data_for(None)
                    filename = f"receipt_{timestamp}_{len(df)+1}.jpg"
                    
                    with st.spinner("Uploading image to Google Drive..."):
                        image_link, image_id = upload_image_to_drive(uploaded_file, filename)
                    
                    if image_link:
                        new_row = {
                            "Ref": len(df) + 1,
                            "Date": formatted_date,
                            "Description": final_desc,
                            "Category": category,
                            "Project Code": project_code,
                            "Project Name": final_project_name,
                            "Amount": final_amt,
                            "Image_Link": image_link,
                            "Image_ID": image_id
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        if save_data_current(df):
                            st.success("Saved successfully!")
                            # Clear cache
                            for key in ["receipt_upload", "receipt_ai_hash", "receipt_ai_data", 
                                       "cap_food_bill_checkbox"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                        else:
                            st.error("Failed to save to Google Sheets")
                    else:
                        st.error("Failed to upload image. Please try again.")

    st.divider()
    st.subheader("📦 Generate Receipts Package")
    if st.button("Generate Receipts ZIP", use_container_width=True):
        try:
            generate_receipts_package(report_month)
        except Exception as e:
            st.error("Failed to generate receipts package.")
            st.exception(e)

# =========================================================
# TAB 2: CREDIT CARD
# =========================================================
with tab2:
    st.subheader("Credit Card")

    cc_view = load_cc_for(None)
    if not cc_view.empty:
        display_columns = ["Date", "Description", "Category", "Project Code"]
        if "Project Name" in cc_view.columns:
            display_columns.append("Project Name")
        display_columns.append("Amount")

        st.dataframe(cc_view[display_columns], use_container_width=True)
        st.metric("Total Credit Card", f"{cc_view['Amount'].sum():.2f} SAR/AED")
    else:
        st.info("No credit card expenses yet.")

    uploaded_statement = st.file_uploader(
        "Upload Credit Card Statement (Camera or File)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        help="📷 Tap to take a photo",
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
                "Hotel Booking", "Food & Beverages", "Visa & Ticket", "Parking",
                "R & D Expenses", "Subscriptions", "Office - Tools & Consumables",
                "Project - Consumables", "Transportation",
                "Project Expenses - Miscellaneous", "Office Expenses - Miscellaneous",
                "Can't classify",
            ]
            cat_index = category_options.index(default_cat) if default_cat in category_options else 7
            category = st.selectbox("Category", category_options, index=cat_index)

            current_month = get_current_month_str(date_input)
            previous_codes = get_project_codes_for_month(month_str=current_month)
            code_options = ["Enter new project code..."] + previous_codes
            selected_code_option = st.selectbox("Project Code", options=code_options, index=0)

            if selected_code_option == "Enter new project code...":
                project_code = st.text_input("Enter New Project Code", 
                                            placeholder="e.g., 250909-PDS-303")
            else:
                project_code = selected_code_option

            project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha",
                                        help="Enter the project name (required)")

        with c2:
            description = st.text_input("Description", value=default_desc)
            amount = st.number_input("Amount (SAR/AED)", min_value=0.0, step=0.5, 
                                    value=float(default_amt))

            cap_to_40 = False
            if category == "Food & Beverages":
                cap_to_40 = st.checkbox("Cap amount to 40 SAR/AED", value=False,
                                       key="cc_cap_food_bill_checkbox")

        submitted = st.form_submit_button("✅ Save Credit Card Expense", use_container_width=True)

        if submitted:
            if not description.strip():
                st.error("Description required.")
            elif not project_name.strip():
                st.error("Project Name is required.")
            else:
                final_desc = description.strip()
                final_amt = float(amount)
                final_project_name = project_name.strip()

                if category == "Food & Beverages" and cap_to_40 and final_amt > 40:
                    final_desc = f"{final_desc} (capped at 40)"
                    final_amt = 40.0
                    st.warning("Amount capped at 40")

                cc_df = load_cc_for(None)
                new_row = {
                    "Date": formatted_date,
                    "Description": final_desc,
                    "Category": category,
                    "Project Code": project_code if project_code else "",
                    "Project Name": final_project_name,
                    "Amount": final_amt,
                }
                cc_df = pd.concat([cc_df, pd.DataFrame([new_row])], ignore_index=True)
                
                if save_cc_current(cc_df):
                    st.success("Saved!")
                    for key in ["cc_upload", "cc_ai_hash", "cc_ai_data", "cc_cap_food_bill_checkbox"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    st.divider()
    st.subheader("📦 Generate Credit Card Package")
    if st.button("Generate Credit Card ZIP", use_container_width=True):
        try:
            generate_credit_card_package(report_month)
        except Exception as e:
            st.error("Failed to generate credit card package.")
            st.exception(e)

# =========================================================
# TAB 3: TRANSPORTATION
# =========================================================
with tab3:
    st.subheader("🚗 Transportation Expenses")

    with st.form("transport_form"):
        col1, col2 = st.columns(2)

        with col1:
            travel_date = st.date_input("Date *", value=datetime.today())
            formatted_date = travel_date.strftime("%d-%b")

            from_location = st.text_input("From *", placeholder="e.g., Dubai, UAE")
            destination = st.text_input("Destination *", placeholder="e.g., Abu Dhabi, UAE")

        with col2:
            return_included = st.checkbox("Return Included")

            current_month = get_current_month_str(travel_date)
            previous_codes = get_project_codes_for_month(month_str=current_month)
            code_options = ["Enter new project code..."] + previous_codes
            selected_code_option = st.selectbox("Project Code (Optional)", 
                                               options=code_options, index=0)

            if selected_code_option == "Enter new project code...":
                project_code = st.text_input("Enter New Project Code", 
                                            placeholder="e.g., 250909-PDS-303")
            else:
                project_code = selected_code_option

            project_name = st.text_input("Project Name *", placeholder="e.g., Project Alpha")

        submitted = st.form_submit_button("✅ Save Transportation Expense", 
                                         use_container_width=True)

        if submitted:
            errors = []
            if not from_location or not from_location.strip():
                errors.append("Please enter From location")
            if not destination or not destination.strip():
                errors.append("Please enter Destination")
            if not project_name or not project_name.strip():
                errors.append("Please enter Project Name")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                transport_df = load_transport_data(None)

                new_row = {
                    "Date": formatted_date,
                    "From": from_location.strip(),
                    "Destination": destination.strip(),
                    "Return Included": return_included,
                    "Project Code": project_code if project_code else "",
                    "Project Name": project_name.strip(),
                }

                transport_df = pd.concat([transport_df, pd.DataFrame([new_row])], 
                                        ignore_index=True)
                
                if save_transport_data(transport_df):
                    st.success("Transportation expense saved!")
                    st.rerun()

    st.divider()
    st.subheader("Current Month Transportation Expenses")
    transport_df = load_transport_data(None)

    if not transport_df.empty:
        display_columns = ["Date", "From", "Destination", "Return Included", "Project Code"]
        if "Project Name" in transport_df.columns:
            display_columns.append("Project Name")

        display_df = transport_df[display_columns].copy()
        display_df["Return Included"] = display_df["Return Included"].apply(
            lambda x: "Yes" if x else "No"
        )

        st.dataframe(display_df, use_container_width=True)

        st.divider()
        st.subheader("Generate Report")
        if st.button("📊 Generate Transportation Excel", use_container_width=True):
            excel_path = "Transportation_Expenses.xlsx"
            generate_transport_excel(transport_df, excel_path)
            st.success("Excel file generated!")
            with open(excel_path, "rb") as fp:
                st.download_button(
                    "⬇️ Download Transportation Expenses Excel",
                    fp,
                    "Transportation_Expenses.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.info("No transport records yet.")
