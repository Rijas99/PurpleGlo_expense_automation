import streamlit as st
import pandas as pd
import os
import shutil
import json
import stat
import re
import time as time_module
import hashlib
import sqlite3
from datetime import datetime, time as dt_time
from PIL import Image
import google.generativeai as genai

# Supabase imports (optional - fallback to SQLite if not available)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# =========================================================
# CONFIGURATION
# =========================================================

APP_VERSION = "3.7.0"  # Version with Supabase cloud database support

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

# Supabase Configuration (for cloud deployment)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
SUPABASE_STORAGE_BUCKET = st.secrets.get("SUPABASE_STORAGE_BUCKET", os.environ.get("SUPABASE_STORAGE_BUCKET", "expense-images"))

# Determine if we should use Supabase (if credentials are provided) or SQLite (local dev)
USE_SUPABASE = SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY

WORK_DIR = "expense_workspace"
CURRENT_DIR = os.path.join(WORK_DIR, "current")
HISTORY_DIR = os.path.join(WORK_DIR, "history")

IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
# Database file - SQLite fallback for local development
DB_FILE = os.path.join(WORK_DIR, "expenses.db")
# Keep CSV file paths for backward compatibility and migration
DATA_FILE = os.path.join(CURRENT_DIR, "data.csv")
CREDIT_CARD_DATA_FILE = os.path.join(CURRENT_DIR, "credit_card_data.csv")
TRANSPORT_DATA_FILE = os.path.join(CURRENT_DIR, "transport_data.csv")

MAX_HISTORY_MONTHS = 3  # ✅ keep only last 3 months

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# =========================================================
# DATABASE FUNCTIONS
# =========================================================

# Initialize Supabase client (cached)
@st.cache_resource
def get_supabase_client():
    """Get Supabase client instance."""
    if not USE_SUPABASE:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None


def get_db_connection():
    """Get SQLite database connection (fallback for local development)."""
    if USE_SUPABASE:
        return None  # Use Supabase instead
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables if they don't exist."""
    if USE_SUPABASE:
        # For Supabase, tables should be created via SQL migration
        # Just ensure storage bucket exists
        supabase = get_supabase_client()
        if supabase:
            try:
                # Try to create storage bucket if it doesn't exist
                buckets = supabase.storage.list_buckets()
                bucket_names = [b.name for b in buckets]
                if SUPABASE_STORAGE_BUCKET not in bucket_names:
                    try:
                        supabase.storage.create_bucket(SUPABASE_STORAGE_BUCKET, {"public": False})
                    except Exception:
                        pass  # Bucket might already exist or creation failed
            except Exception:
                pass  # Storage might not be available or configured
        return
    
    # SQLite initialization (local development)
    conn = get_db_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    
    # Receipts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ref INTEGER NOT NULL,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            project_code TEXT,
            project_name TEXT NOT NULL,
            amount REAL NOT NULL,
            original_image_path TEXT,
            month_slug TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Credit card table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_card (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            project_code TEXT,
            project_name TEXT NOT NULL,
            amount REAL NOT NULL,
            month_slug TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Transport table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transport (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            from_location TEXT NOT NULL,
            destination TEXT NOT NULL,
            return_included INTEGER NOT NULL DEFAULT 0,
            project_code TEXT,
            project_name TEXT NOT NULL,
            month_slug TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_month ON receipts(month_slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_receipts_ref ON receipts(ref)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cc_month ON credit_card(month_slug)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transport_month ON transport(month_slug)")
    
    conn.commit()
    conn.close()


def migrate_csv_to_db():
    """Migrate existing CSV data to database if CSV files exist but DB is empty."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return
        
        # Check if database has any data
        try:
            receipts_result = supabase.table("receipts").select("id", count="exact").limit(1).execute()
            receipts_count = receipts_result.count if hasattr(receipts_result, 'count') else len(receipts_result.data) if receipts_result.data else 0
            
            cc_result = supabase.table("credit_card").select("id", count="exact").limit(1).execute()
            cc_count = cc_result.count if hasattr(cc_result, 'count') else len(cc_result.data) if cc_result.data else 0
            
            transport_result = supabase.table("transport").select("id", count="exact").limit(1).execute()
            transport_count = transport_result.count if hasattr(transport_result, 'count') else len(transport_result.data) if transport_result.data else 0
        except Exception:
            return  # Tables might not exist yet
        
        # Only migrate if DB is empty and CSV files exist
        if receipts_count == 0 and os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
                if not df.empty:
                    records = []
                    for _, row in df.iterrows():
                        records.append({
                            "ref": int(row.get("Ref", 0)),
                            "date": str(row.get("Date", "")),
                            "description": str(row.get("Description", "")),
                            "category": str(row.get("Category", "")),
                            "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                            "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                            "amount": float(row.get("Amount", 0)),
                            "original_image_path": str(row.get("Original_Image_Path", "")) if pd.notna(row.get("Original_Image_Path")) else "",
                            "month_slug": None
                        })
                    if records:
                        supabase.table("receipts").insert(records).execute()
            except Exception:
                pass
        
        if cc_count == 0 and os.path.exists(CREDIT_CARD_DATA_FILE):
            try:
                df = pd.read_csv(CREDIT_CARD_DATA_FILE)
                if not df.empty:
                    records = []
                    for _, row in df.iterrows():
                        records.append({
                            "date": str(row.get("Date", "")),
                            "description": str(row.get("Description", "")),
                            "category": str(row.get("Category", "")),
                            "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                            "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                            "amount": float(row.get("Amount", 0)),
                            "month_slug": None
                        })
                    if records:
                        supabase.table("credit_card").insert(records).execute()
            except Exception:
                pass
        
        if transport_count == 0 and os.path.exists(TRANSPORT_DATA_FILE):
            try:
                df = pd.read_csv(TRANSPORT_DATA_FILE)
                if not df.empty:
                    records = []
                    for _, row in df.iterrows():
                        records.append({
                            "date": str(row.get("Date", "")),
                            "from_location": str(row.get("From", "")),
                            "destination": str(row.get("Destination", "")),
                            "return_included": 1 if row.get("Return Included", False) else 0,
                            "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                            "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                            "month_slug": None
                        })
                    if records:
                        supabase.table("transport").insert(records).execute()
            except Exception:
                pass
        return
    
    # SQLite migration (local development)
    conn = get_db_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    
    # Check if database has any data
    cursor.execute("SELECT COUNT(*) FROM receipts")
    receipts_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM credit_card")
    cc_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM transport")
    transport_count = cursor.fetchone()[0]
    
    # Only migrate if DB is empty and CSV files exist
    if receipts_count == 0 and os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO receipts (ref, date, description, category, project_code, project_name, amount, original_image_path, month_slug)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        int(row.get("Ref", 0)),
                        str(row.get("Date", "")),
                        str(row.get("Description", "")),
                        str(row.get("Category", "")),
                        str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                        str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                        float(row.get("Amount", 0)),
                        str(row.get("Original_Image_Path", "")) if pd.notna(row.get("Original_Image_Path")) else "",
                        None  # Current month
                    ))
                conn.commit()
        except Exception as e:
            pass  # Silently fail migration if CSV is corrupted
    
    if cc_count == 0 and os.path.exists(CREDIT_CARD_DATA_FILE):
        try:
            df = pd.read_csv(CREDIT_CARD_DATA_FILE)
            if not df.empty:
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO credit_card (date, description, category, project_code, project_name, amount, month_slug)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(row.get("Date", "")),
                        str(row.get("Description", "")),
                        str(row.get("Category", "")),
                        str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                        str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                        float(row.get("Amount", 0)),
                        None  # Current month
                    ))
                conn.commit()
        except Exception as e:
            pass  # Silently fail migration if CSV is corrupted
    
    if transport_count == 0 and os.path.exists(TRANSPORT_DATA_FILE):
        try:
            df = pd.read_csv(TRANSPORT_DATA_FILE)
            if not df.empty:
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO transport (date, from_location, destination, return_included, project_code, project_name, month_slug)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(row.get("Date", "")),
                        str(row.get("From", "")),
                        str(row.get("Destination", "")),
                        1 if row.get("Return Included", False) else 0,
                        str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                        str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                        None  # Current month
                    ))
                conn.commit()
        except Exception as e:
            pass  # Silently fail migration if CSV is corrupted
    
    conn.close()


# Initialize database on import
init_database()
migrate_csv_to_db()


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
    """List archived months from database (Supabase) or local folders (SQLite)."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return []
        
        try:
            # Get distinct month_slug values from all three tables
            months_set = set()
            
            # Query receipts table - get all records and filter for non-null month_slug
            receipts_response = supabase.table("receipts").select("month_slug").execute()
            if receipts_response.data:
                for record in receipts_response.data:
                    month_slug = record.get("month_slug")
                    if month_slug and month_slug.strip():  # Non-null and non-empty
                        months_set.add(month_slug)
            
            # Query credit_card table
            cc_response = supabase.table("credit_card").select("month_slug").execute()
            if cc_response.data:
                for record in cc_response.data:
                    month_slug = record.get("month_slug")
                    if month_slug and month_slug.strip():
                        months_set.add(month_slug)
            
            # Query transport table
            transport_response = supabase.table("transport").select("month_slug").execute()
            if transport_response.data:
                for record in transport_response.data:
                    month_slug = record.get("month_slug")
                    if month_slug and month_slug.strip():
                        months_set.add(month_slug)
            
            months = list(months_set)
            
            # Sort by parsing month_slug (e.g., "Jan_2026" -> datetime)
            def parse_month_slug(slug):
                try:
                    # Format: "Jan_2026" -> datetime(2026, 1, 1)
                    parts = slug.split("_")
                    if len(parts) == 2:
                        month_str, year_str = parts
                        month_map = {
                            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                        }
                        month = month_map.get(month_str, 1)
                        year = int(year_str)
                        return datetime(year, month, 1)
                except:
                    # If parsing fails, return a very old date to put it at the end
                    return datetime(1900, 1, 1)
            
            months.sort(key=parse_month_slug, reverse=True)
            return months
        except Exception:
            # Fallback to local folders if Supabase query fails
            pass
    
    # Local folder check (SQLite mode or Supabase fallback)
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
    Archive current month data in database by updating month_slug, then copy images to archive folder.
    Old archived data (older than 3 months) is kept in the database but separated by month_slug
    to prevent mixing with current month data. Local archive folders may be cleaned up manually if needed.
    """
    ensure_current_dirs()
    month_slug = slugify(report_month)  # e.g. Jan_2026
    target = month_folder_path(month_slug)

    if os.path.exists(target):
        raise RuntimeError(
            f"Archive folder already exists: {month_slug}\n"
            f"Change Report Month or delete that archive folder."
        )

    # Check if there's data to archive
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            raise RuntimeError("Database connection failed.")
        
        try:
            receipts_result = supabase.table("receipts").select("id", count="exact").is_("month_slug", "null").limit(1).execute()
            receipts_count = receipts_result.count if hasattr(receipts_result, 'count') else len(receipts_result.data) if receipts_result.data else 0
            
            cc_result = supabase.table("credit_card").select("id", count="exact").is_("month_slug", "null").limit(1).execute()
            cc_count = cc_result.count if hasattr(cc_result, 'count') else len(cc_result.data) if cc_result.data else 0
            
            transport_result = supabase.table("transport").select("id", count="exact").is_("month_slug", "null").limit(1).execute()
            transport_count = transport_result.count if hasattr(transport_result, 'count') else len(transport_result.data) if transport_result.data else 0
        except Exception:
            receipts_count = cc_count = transport_count = 0
        
        if receipts_count == 0 and cc_count == 0 and transport_count == 0:
            raise RuntimeError("No data to archive. Current month is empty.")
        
        # Update month_slug in database for all current month records
        try:
            supabase.table("receipts").update({"month_slug": month_slug}).is_("month_slug", "null").execute()
            supabase.table("credit_card").update({"month_slug": month_slug}).is_("month_slug", "null").execute()
            supabase.table("transport").update({"month_slug": month_slug}).is_("month_slug", "null").execute()
        except Exception as e:
            raise RuntimeError(f"Error archiving data: {e}")
    else:
        conn = get_db_connection()
        if conn is None:
            raise RuntimeError("Database connection failed.")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM receipts WHERE month_slug IS NULL")
        receipts_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM credit_card WHERE month_slug IS NULL")
        cc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM transport WHERE month_slug IS NULL")
        transport_count = cursor.fetchone()[0]
        
        if receipts_count == 0 and cc_count == 0 and transport_count == 0:
            conn.close()
            raise RuntimeError("No data to archive. Current month is empty.")
        
        # Update month_slug in database for all current month records
        cursor.execute("UPDATE receipts SET month_slug = ? WHERE month_slug IS NULL", (month_slug,))
        cursor.execute("UPDATE credit_card SET month_slug = ? WHERE month_slug IS NULL", (month_slug,))
        cursor.execute("UPDATE transport SET month_slug = ? WHERE month_slug IS NULL", (month_slug,))
        conn.commit()
        conn.close()

    # create target folder for images
    os.makedirs(target, exist_ok=True)
    os.makedirs(os.path.join(target, "images"), exist_ok=True)

    # Export CSV files for backup (optional, for compatibility)
    try:
        receipts_df = load_data_for(month_slug)
        if not receipts_df.empty:
            receipts_df.to_csv(os.path.join(target, "data.csv"), index=False)
        
        cc_df = load_cc_for(month_slug)
        if not cc_df.empty:
            cc_df.to_csv(os.path.join(target, "credit_card_data.csv"), index=False)
        
        transport_df = load_transport_data(month_slug)
        if not transport_df.empty:
            transport_df.to_csv(os.path.join(target, "transport_data.csv"), index=False)
    except Exception:
        pass  # CSV export is optional

    # copy images to archive
    if os.path.exists(IMAGES_DIR):
        # Get list of images that belong to archived receipts
        receipts_df = load_data_for(month_slug)
        archived_images = set()
        if "Original_Image_Path" in receipts_df.columns:
            archived_images = set(receipts_df["Original_Image_Path"].dropna().astype(str))
        
        # Copy archived images
        for img_file in archived_images:
            src = os.path.join(IMAGES_DIR, img_file)
            if os.path.exists(src):
                dst = os.path.join(target, "images", img_file)
                try:
                    shutil.copy(src, dst)
                except Exception:
                    pass
        
        # Clean up current images directory (keep only non-archived images)
        # Actually, let's keep all images for now to avoid accidental deletion
        # Images will be cleaned up naturally over time

    # Clear current month data (images directory stays, but data is archived)
    # Note: We don't delete CURRENT_DIR anymore since we're using database
    # Just ensure images directory exists for new month
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Note: Old archived data is kept in the database (separated by month_slug)
    # to prevent mixing with current month. No automatic cleanup is performed.
    # The trim_history_keep_last_n() function is available for manual cleanup if needed.


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
    """Load receipts data from database."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame(columns=["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount", "Original_Image_Path"])
        
        try:
            if selected_month_slug:
                response = supabase.table("receipts").select("*").eq("month_slug", selected_month_slug).order("ref").execute()
            else:
                response = supabase.table("receipts").select("*").is_("month_slug", "null").order("ref").execute()
            
            if not response.data:
                return pd.DataFrame(columns=["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount", "Original_Image_Path"])
            
            df = pd.DataFrame(response.data)
        except Exception:
            return pd.DataFrame(columns=["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount", "Original_Image_Path"])
    else:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame(columns=["Ref", "Date", "Description", "Category", "Project Code", "Project Name", "Amount", "Original_Image_Path"])
        
        if selected_month_slug:
            query = "SELECT * FROM receipts WHERE month_slug = ? ORDER BY ref"
            df = pd.read_sql_query(query, conn, params=(selected_month_slug,))
        else:
            query = "SELECT * FROM receipts WHERE month_slug IS NULL ORDER BY ref"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Ref",
                "Date",
                "Description",
                "Category",
                "Project Code",
                "Project Name",
                "Amount",
                "Original_Image_Path",
            ]
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "ref": "Ref",
        "date": "Date",
        "description": "Description",
        "category": "Category",
        "project_code": "Project Code",
        "project_name": "Project Name",
        "amount": "Amount",
        "original_image_path": "Original_Image_Path"
    })
    
    # Ensure Ref is integer
    if "Ref" in df.columns:
        df["Ref"] = df["Ref"].astype(int)
    
    return df


def save_data_current(df: pd.DataFrame):
    """Save receipts data to database (current month)."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return
        
        try:
            # Delete existing current month data
            supabase.table("receipts").delete().is_("month_slug", "null").execute()
            
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                records.append({
                    "ref": int(row.get("Ref", 0)),
                    "date": str(row.get("Date", "")),
                    "description": str(row.get("Description", "")),
                    "category": str(row.get("Category", "")),
                    "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                    "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                    "amount": float(row.get("Amount", 0)),
                    "original_image_path": str(row.get("Original_Image_Path", "")) if pd.notna(row.get("Original_Image_Path")) else "",
                    "month_slug": None
                })
            
            # Insert new data
            if records:
                supabase.table("receipts").insert(records).execute()
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")
    else:
        conn = get_db_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        
        # Delete existing current month data
        cursor.execute("DELETE FROM receipts WHERE month_slug IS NULL")
        
        # Insert new data
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO receipts (ref, date, description, category, project_code, project_name, amount, original_image_path, month_slug)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row.get("Ref", 0)),
                str(row.get("Date", "")),
                str(row.get("Description", "")),
                str(row.get("Category", "")),
                str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                float(row.get("Amount", 0)),
                str(row.get("Original_Image_Path", "")) if pd.notna(row.get("Original_Image_Path")) else "",
                None  # Current month
            ))
        
        conn.commit()
        conn.close()
    
    # Also save to CSV for backup compatibility
    try:
        df.to_csv(DATA_FILE, index=False)
    except Exception:
        pass


def delete_receipt(ref: int) -> tuple[bool, str]:
    """
    Delete a receipt by its Ref number.
    Removes the row from the database and deletes the associated image file.
    Also updates the CSV backup file to keep it in sync.
    Returns (success: bool, message: str)
    """
    try:
        if USE_SUPABASE:
            supabase = get_supabase_client()
            if not supabase:
                return False, "Database connection failed."
            
            # Find the receipt with matching Ref in current month
            response = supabase.table("receipts").select("original_image_path").eq("ref", ref).is_("month_slug", "null").execute()
            
            if not response.data:
                return False, f"Receipt with Ref {ref} not found."
            
            image_path = response.data[0].get("original_image_path") if response.data[0].get("original_image_path") else None
            
            # Delete from database
            supabase.table("receipts").delete().eq("ref", ref).is_("month_slug", "null").execute()
            
            # Delete image from Supabase Storage if it exists
            if image_path:
                try:
                    supabase.storage.from_(SUPABASE_STORAGE_BUCKET).remove([image_path])
                except Exception:
                    pass  # Image might not exist in storage
        else:
            conn = get_db_connection()
            if conn is None:
                return False, "Database connection failed."
            cursor = conn.cursor()
            
            # Find the receipt with matching Ref in current month
            cursor.execute("SELECT original_image_path FROM receipts WHERE ref = ? AND month_slug IS NULL", (ref,))
            result = cursor.fetchone()
            
            if result is None:
                conn.close()
                return False, f"Receipt with Ref {ref} not found."

            # Get the image path before deleting
            image_path = result[0] if result[0] else None

            # Delete from database
            cursor.execute("DELETE FROM receipts WHERE ref = ? AND month_slug IS NULL", (ref,))
            conn.commit()
            conn.close()

            # Delete the associated image file if it exists
            if image_path:
                full_image_path = os.path.join(IMAGES_DIR, str(image_path))
                if os.path.exists(full_image_path):
                    try:
                        os.remove(full_image_path)
                    except Exception:
                        pass
        
        # Update CSV backup file to keep it in sync with database
        try:
            # Reload current data from database (after deletion)
            remaining_df = load_data_for(None)
            # Save to CSV to keep backup in sync
            remaining_df.to_csv(DATA_FILE, index=False)
        except Exception:
            pass  # CSV update is optional, don't fail if it errors

        return True, f"Receipt {ref} deleted successfully."

    except Exception as e:
        return False, f"Error deleting receipt: {str(e)}"


def load_cc_for(selected_month_slug: str | None):
    """Load credit card data from database."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame(columns=["Date", "Description", "Category", "Project Code", "Project Name", "Amount"])
        
        try:
            if selected_month_slug:
                response = supabase.table("credit_card").select("*").eq("month_slug", selected_month_slug).order("date").execute()
            else:
                response = supabase.table("credit_card").select("*").is_("month_slug", "null").order("date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=["Date", "Description", "Category", "Project Code", "Project Name", "Amount"])
            
            df = pd.DataFrame(response.data)
        except Exception:
            return pd.DataFrame(columns=["Date", "Description", "Category", "Project Code", "Project Name", "Amount"])
    else:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame(columns=["Date", "Description", "Category", "Project Code", "Project Name", "Amount"])
        
        if selected_month_slug:
            query = "SELECT * FROM credit_card WHERE month_slug = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(selected_month_slug,))
        else:
            query = "SELECT * FROM credit_card WHERE month_slug IS NULL ORDER BY date"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Description",
                "Category",
                "Project Code",
                "Project Name",
                "Amount",
            ]
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "date": "Date",
        "description": "Description",
        "category": "Category",
        "project_code": "Project Code",
        "project_name": "Project Name",
        "amount": "Amount"
    })
    
    return df


def save_cc_current(df: pd.DataFrame):
    """Save credit card data to database (current month)."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return
        
        try:
            # Delete existing current month data
            supabase.table("credit_card").delete().is_("month_slug", "null").execute()
            
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                records.append({
                    "date": str(row.get("Date", "")),
                    "description": str(row.get("Description", "")),
                    "category": str(row.get("Category", "")),
                    "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                    "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                    "amount": float(row.get("Amount", 0)),
                    "month_slug": None
                })
            
            # Insert new data
            if records:
                supabase.table("credit_card").insert(records).execute()
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")
    else:
        conn = get_db_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        
        # Delete existing current month data
        cursor.execute("DELETE FROM credit_card WHERE month_slug IS NULL")
        
        # Insert new data
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO credit_card (date, description, category, project_code, project_name, amount, month_slug)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("Date", "")),
                str(row.get("Description", "")),
                str(row.get("Category", "")),
                str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                float(row.get("Amount", 0)),
                None  # Current month
            ))
        
        conn.commit()
        conn.close()
    
    # Also save to CSV for backup compatibility
    try:
        df.to_csv(CREDIT_CARD_DATA_FILE, index=False)
    except Exception:
        pass


def load_transport_data(selected_month_slug: str | None = None):
    """Load transport data from database for current or archived month."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame(columns=["Date", "From", "Destination", "Return Included", "Project Code", "Project Name"])
        
        try:
            if selected_month_slug:
                response = supabase.table("transport").select("*").eq("month_slug", selected_month_slug).order("date").execute()
            else:
                response = supabase.table("transport").select("*").is_("month_slug", "null").order("date").execute()
            
            if not response.data:
                return pd.DataFrame(columns=["Date", "From", "Destination", "Return Included", "Project Code", "Project Name"])
            
            df = pd.DataFrame(response.data)
        except Exception:
            return pd.DataFrame(columns=["Date", "From", "Destination", "Return Included", "Project Code", "Project Name"])
    else:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame(columns=["Date", "From", "Destination", "Return Included", "Project Code", "Project Name"])
        
        if selected_month_slug:
            query = "SELECT * FROM transport WHERE month_slug = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(selected_month_slug,))
        else:
            query = "SELECT * FROM transport WHERE month_slug IS NULL ORDER BY date"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "From",
                "Destination",
                "Return Included",
                "Project Code",
                "Project Name",
            ]
        )
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "date": "Date",
        "from_location": "From",
        "destination": "Destination",
        "return_included": "Return Included",
        "project_code": "Project Code",
        "project_name": "Project Name"
    })
    
    # Convert return_included from int to boolean
    if "Return Included" in df.columns:
        df["Return Included"] = df["Return Included"].astype(bool)
    
    return df


def save_transport_data(df: pd.DataFrame):
    """Save transport data to database (current month)."""
    if USE_SUPABASE:
        supabase = get_supabase_client()
        if not supabase:
            return
        
        try:
            # Delete existing current month data
            supabase.table("transport").delete().is_("month_slug", "null").execute()
            
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                records.append({
                    "date": str(row.get("Date", "")),
                    "from_location": str(row.get("From", "")),
                    "destination": str(row.get("Destination", "")),
                    "return_included": 1 if row.get("Return Included", False) else 0,
                    "project_code": str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                    "project_name": str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                    "month_slug": None
                })
            
            # Insert new data
            if records:
                supabase.table("transport").insert(records).execute()
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")
    else:
        conn = get_db_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        
        # Delete existing current month data
        cursor.execute("DELETE FROM transport WHERE month_slug IS NULL")
        
        # Insert new data
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO transport (date, from_location, destination, return_included, project_code, project_name, month_slug)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("Date", "")),
                str(row.get("From", "")),
                str(row.get("Destination", "")),
                1 if row.get("Return Included", False) else 0,
                str(row.get("Project Code", "")) if pd.notna(row.get("Project Code")) else "",
                str(row.get("Project Name", "")) if pd.notna(row.get("Project Name")) else "",
                None  # Current month
            ))
        
        conn.commit()
        conn.close()
    
    # Also save to CSV for backup compatibility
    try:
        df.to_csv(TRANSPORT_DATA_FILE, index=False)
    except Exception:
        pass


def generate_transport_excel(df: pd.DataFrame, output_path: str):
    """
    Generate Excel file for transport expenses.
    Columns: Date, From, Destination, Return Included, Project Code, Project Name
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
    if "Project Name" in df.columns:
        excel_df["Project Name"] = df["Project Name"].fillna("")

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
    if not text:
        raise ValueError("Empty response from Gemini.")

    # Remove markdown code blocks
    text = text.strip().replace("```json", "").replace("```", "").strip()

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try a greedy approach - find content between first { and last }
    # This handles nested objects and objects with arrays
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace : last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object patterns (handle simple nested cases)
    # Look for { followed by content and }
    json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Try each match, starting with the longest (most likely to be complete)
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # If all else fails, raise with the actual response for debugging
    raise ValueError(
        f"No valid JSON object found in Gemini response. Response: {text[:200]}"
    )


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

    # Optimize for speed: temperature=0 for faster deterministic responses,
    # max_output_tokens set to allow for complete JSON responses (increased slightly for reliability)
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 1000,  # Increased to ensure complete JSON responses
    }

    try:
        response = model.generate_content(
            [prompt, img], generation_config=generation_config
        )

        if not response or not response.text:
            st.error("AI Error: Empty response from Gemini API.")
            return None

        return _extract_json(response.text)
    except ValueError as e:
        # JSON extraction error - show helpful message
        error_msg = str(e)
        st.error(f"AI Error: Failed to parse JSON from Gemini response. {error_msg}")
        # Log the raw response for debugging (first 500 chars)
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
                response = model.generate_content(
                    [prompt, img], generation_config=generation_config
                )
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
            image_path = str(row["Original_Image_Path"])
            dst = os.path.join(bills_folder, f"{row['Ref']}.jpg")
            
            if USE_SUPABASE:
                supabase = get_supabase_client()
                if supabase:
                    try:
                        # Download from Supabase Storage
                        file_data = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(image_path)
                        with open(dst, "wb") as f:
                            f.write(file_data)
                    except Exception:
                        # Fallback to local if Supabase download fails
                        src = os.path.join(images_dir, image_path)
                        if os.path.exists(src):
                            shutil.copy(src, dst)
            else:
                # Local storage
                src = os.path.join(images_dir, image_path)
                if os.path.exists(src):
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
st.caption(f"Version {APP_VERSION}")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown(f"**🟣 Version {APP_VERSION}**")
st.sidebar.markdown("---")
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
        # Check if Project Name column exists (for backward compatibility)
        display_columns = ["Ref", "Date", "Description", "Category", "Project Code"]
        if "Project Name" in df_view.columns:
            display_columns.append("Project Name")
        display_columns.append("Amount")

        # Create custom table with inline delete buttons
        # Determine column layout based on whether Project Name exists
        has_project_name = "Project Name" in df_view.columns
        num_cols = 8 if has_project_name else 7  # Ref, Date, Description, Category, Project Code, [Project Name], Amount, Delete
        
        # Create header row
        header_cols = st.columns(num_cols)
        header_cols[0].markdown("**Ref**")
        header_cols[1].markdown("**Date**")
        header_cols[2].markdown("**Description**")
        header_cols[3].markdown("**Category**")
        header_cols[4].markdown("**Project Code**")
        col_idx = 5
        if has_project_name:
            header_cols[col_idx].markdown("**Project Name**")
            col_idx += 1
        header_cols[col_idx].markdown("**Amount**")
        header_cols[col_idx + 1].markdown("**Delete**")
        
        st.divider()
        
        # Create data rows with inline delete buttons
        for idx, row in df_view.iterrows():
            row_cols = st.columns(num_cols)
            ref_value = int(row["Ref"])
            
            # Format data for display
            description = str(row["Description"])
            if len(description) > 35:
                description = description[:35] + "..."
            
            category = str(row["Category"])
            if len(category) > 22:
                category = category[:22] + "..."
            
            project_code = str(row["Project Code"]) if pd.notna(row["Project Code"]) else ""
            if len(project_code) > 18:
                project_code = project_code[:18] + "..."
            
            amount = f"{float(row['Amount']):.2f}"
            
            # Display data in columns
            row_cols[0].write(str(ref_value))
            row_cols[1].write(str(row["Date"]))
            row_cols[2].write(description)
            row_cols[3].write(category)
            row_cols[4].write(project_code)
            col_idx = 5
            if has_project_name:
                project_name = str(row["Project Name"]) if pd.notna(row["Project Name"]) else ""
                if len(project_name) > 20:
                    project_name = project_name[:20] + "..."
                row_cols[col_idx].write(project_name)
                col_idx += 1
            row_cols[col_idx].write(amount)
            
            # Delete button in last column (only for CURRENT month)
            if month_view == "CURRENT":
                with row_cols[col_idx + 1]:
                    if st.button(
                        "🗑️",
                        key=f"delete_receipt_{ref_value}",
                        help=f"Delete receipt {ref_value}: {row['Description'][:25]}",
                        use_container_width=True,
                    ):
                        success, message = delete_receipt(ref_value)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            else:
                row_cols[col_idx + 1].write("—")  # Show dash for archived months

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

            # Automatically analyze receipt if not already cached
            if st.session_state.get("receipt_ai_hash") == h and st.session_state.get(
                "receipt_ai_data"
            ):
                # Use cached result
                ai_data = st.session_state.get("receipt_ai_data", {})
            else:
                # Automatically trigger analysis
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

                    project_name = st.text_input(
                        "Project Name *",
                        placeholder="e.g., Project Alpha",
                        help="Enter the project name (required)",
                    )

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
                    final_project_name = project_name.strip() if project_name else ""

                    # Validation
                    if not final_project_name:
                        st.error(
                            "Project Name is required. Please enter a project name."
                        )
                    else:
                        # Only cap if checkbox is checked, category is Food & Beverages, and amount > 40
                        if (
                            category == "Food & Beverages"
                            and cap_to_40
                            and final_amt > 40
                        ):
                            final_desc = f"{final_desc} (capped at 40)"
                            final_amt = 40.0
                            st.warning("Amount capped at 40")

                        temp_filename = f"temp_{datetime.now().timestamp()}.jpg"
                        
                        # Save image to Supabase Storage or local filesystem
                        if USE_SUPABASE:
                            supabase = get_supabase_client()
                            if supabase:
                                try:
                                    # Upload to Supabase Storage - use getbuffer() for mobile compatibility
                                    # getbuffer() returns memoryview that doesn't consume the stream
                                    file_buffer = uploaded_file.getbuffer()
                                    # Convert to bytes for Supabase (which expects bytes)
                                    file_data = bytes(file_buffer)
                                    
                                    # Upload with correct file options format
                                    response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
                                        temp_filename,
                                        file_data,
                                        file_options={"contentType": "image/jpeg", "upsert": "true"}
                                    )
                                    # Check if upload was successful
                                    if response:
                                        st.success("Image uploaded to Supabase Storage!")
                                except Exception as e:
                                    st.error(f"Error uploading image to Supabase: {e}")
                                    st.exception(e)  # Show full error for debugging
                                    # Fallback to local storage - use getbuffer() for mobile compatibility
                                    file_buffer = uploaded_file.getbuffer()
                                    save_path = os.path.join(IMAGES_DIR, temp_filename)
                                    with open(save_path, "wb") as f:
                                        f.write(bytes(file_buffer))
                        else:
                            # Local storage (SQLite mode) - use getbuffer() for mobile compatibility
                            file_buffer = uploaded_file.getbuffer()
                            save_path = os.path.join(IMAGES_DIR, temp_filename)
                            with open(save_path, "wb") as f:
                                f.write(bytes(file_buffer))

                        df = load_data_for(None)
                        # Calculate next Ref number: use max(Ref) + 1 if df is not empty, otherwise start at 1
                        next_ref = int(df["Ref"].max()) + 1 if not df.empty and "Ref" in df.columns else 1
                        new_row = {
                            "Ref": next_ref,
                            "Date": formatted_date,
                            "Description": final_desc,
                            "Category": category,
                            "Project Code": project_code,
                            "Project Name": final_project_name,
                            "Amount": final_amt,
                            "Original_Image_Path": temp_filename,
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        save_data_current(df)
                        st.success("Saved!")

                        # Clear form and photo after successful save
                        if "receipt_upload" in st.session_state:
                            del st.session_state["receipt_upload"]
                        if "receipt_ai_hash" in st.session_state:
                            del st.session_state["receipt_ai_hash"]
                        if "receipt_ai_data" in st.session_state:
                            del st.session_state["receipt_ai_data"]
                        if "cap_food_bill_checkbox" in st.session_state:
                            del st.session_state["cap_food_bill_checkbox"]

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
        # Prepare display columns with backward compatibility
        display_columns = ["Date", "Description", "Category", "Project Code"]
        if "Project Name" in cc_view.columns:
            display_columns.append("Project Name")
        display_columns.append("Amount")

        st.dataframe(
            cc_view[display_columns],
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

            # Automatically analyze statement if not already cached
            if st.session_state.get("cc_ai_hash") == h and st.session_state.get(
                "cc_ai_data"
            ):
                # Use cached result
                ai_data = st.session_state.get("cc_ai_data", {})
            else:
                # Automatically trigger analysis
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

                project_name = st.text_input(
                    "Project Name *",
                    placeholder="e.g., Project Alpha",
                    help="Enter the project name (required)",
                )

            with c2:
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
                        key="cc_cap_food_bill_checkbox",
                        help="If checked, amount will be capped at 40 and noted in description",
                    )

            submitted = st.form_submit_button(
                "✅ Save Credit Card Expense", use_container_width=True
            )

            if submitted:
                if not description.strip():
                    st.error("Description required.")
                elif not project_name.strip():
                    st.error("Project Name is required. Please enter a project name.")
                else:
                    final_desc = description.strip()
                    final_amt = float(amount)
                    final_project_name = project_name.strip()

                    # Only cap if checkbox is checked, category is Food & Beverages, and amount > 40
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
                    cc_df = pd.concat(
                        [cc_df, pd.DataFrame([new_row])], ignore_index=True
                    )
                    save_cc_current(cc_df)
                    st.success("Saved!")

                    # Clear form and photo after successful save
                    if "cc_upload" in st.session_state:
                        del st.session_state["cc_upload"]
                    if "cc_ai_hash" in st.session_state:
                        del st.session_state["cc_ai_hash"]
                    if "cc_ai_data" in st.session_state:
                        del st.session_state["cc_ai_data"]
                    if "cc_cap_food_bill_checkbox" in st.session_state:
                        del st.session_state["cc_cap_food_bill_checkbox"]

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

            project_name = st.text_input(
                "Project Name *",
                placeholder="e.g., Project Alpha",
                help="Enter the project name (required)",
            )

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
            if not project_name or not project_name.strip():
                errors.append("Please enter Project Name")

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
                    "Project Name": project_name.strip(),
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
            # Prepare display dataframe with backward compatibility
            display_columns = [
                "Date",
                "From",
                "Destination",
                "Return Included",
                "Project Code",
            ]
            if "Project Name" in month_transport.columns:
                display_columns.append("Project Name")

            display_df = month_transport[display_columns].copy()

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

            # Handle empty/null project names
            if "Project Name" in display_df.columns:
                display_df["Project Name"] = (
                    display_df["Project Name"].fillna("").astype(str)
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
