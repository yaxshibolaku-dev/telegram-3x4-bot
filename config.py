import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN topilmadi! .env faylini tekshiring.")

# Gemini Vision AI settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip()

# Directory paths
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Print / Image Dimensions at 300 DPI (Dots Per Inch)
# 1 inch = 2.54 cm -> DPI / 2.54 = ~118.11 pixels per cm
DPI = 300
CM_TO_PX = DPI / 2.54

# 3x4 cm Photo Size in Pixels
PHOTO_WIDTH_CM = 3.0
PHOTO_HEIGHT_CM = 4.0
PHOTO_WIDTH_PX = int(round(PHOTO_WIDTH_CM * CM_TO_PX))   # 354 px
PHOTO_HEIGHT_PX = int(round(PHOTO_HEIGHT_CM * CM_TO_PX)) # 472 px

# 10x15 cm Paper Size in Pixels (Landscape: 15cm width, 10cm height)
SHEET_WIDTH_CM = 15.0
SHEET_HEIGHT_CM = 10.0
SHEET_WIDTH_PX = int(round(SHEET_WIDTH_CM * CM_TO_PX))   # 1772 px
SHEET_HEIGHT_PX = int(round(SHEET_HEIGHT_CM * CM_TO_PX)) # 1181 px

# Grid configuration: 4 columns x 2 rows = 8 photos
GRID_COLS = 4
GRID_ROWS = 2
PHOTO_COUNT = GRID_COLS * GRID_ROWS # 8
