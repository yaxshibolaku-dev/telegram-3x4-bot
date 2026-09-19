import io
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps, ImageDraw
import numpy as np

from config import (
    BASE_DIR,
    PHOTO_WIDTH_PX,
    PHOTO_HEIGHT_PX,
    SHEET_WIDTH_PX,
    SHEET_HEIGHT_PX,
    GRID_COLS,
    GRID_ROWS,
    DPI
)

SUIT_DIR = BASE_DIR / "kostyum oq ko'ylak erkak"

# Kostyum va ko'ylak shablonlari katalogi
SUIT_CATALOG = {
    "suit_blue": {
        "title": "To‘q ko‘k kostyum (galstukli)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_01 PM (1).png",
        "collar_y_ratio": 0.38,  # Yoqa pastki markazi
    },
    "suit_black_red": {
        "title": "Qora kostyum (qizil galstuk)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (2).png",
        "collar_y_ratio": 0.40,
    },
    "suit_grey": {
        "title": "To‘q kulrang kostyum (ko‘k galstuk)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (3).png",
        "collar_y_ratio": 0.42,
    },
    "suit_black": {
        "title": "Qora kostyum (galstuksiz)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (4).png",
        "collar_y_ratio": 0.44,
    },
    "shirt_white": {
        "title": "Oq klassik ko‘ylak",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_38_55 PM (2).png",
        "collar_y_ratio": 0.42,
    }
}


def load_suit_template(suit_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Tanlangan kostyum shablonini yuklaydi.
    """
    info = SUIT_CATALOG.get(suit_id, SUIT_CATALOG["suit_blue"])
    suit_file_path = SUIT_DIR / info["file"]
    if not suit_file_path.exists():
        raise FileNotFoundError(f"Kostyum shabloni topilmadi: {suit_file_path}")
    
    suit_img = Image.open(suit_file_path).convert("RGBA")
    return suit_img, info


def overlay_suit(
    user_person_rgba: Image.Image,
    head_box: Optional[list] = None,
    suit_id: str = "suit_blue"
) -> Image.Image:
    """
    Foydalanuvchining fonsiz (RGBA) fotosuratiga kostyumni moslashtirib ustiga kiygizadi.
    Natijada insonning boshi va bo'yni kostyum yoqasi ichidan tabiiy chiqib turadi,
    kostyum esa pastki kiyimlarni to'liq yopadi.
    """
    suit_img, info = load_suit_template(suit_id)
    suit_w, suit_h = suit_img.size  # 1086 x 1448

    # Foydalanuvchi rasmini exif bo'yicha to'g'rilash
    user_person = ImageOps.exif_transpose(user_person_rgba)
    if user_person.mode != "RGBA":
        user_person = user_person.convert("RGBA")

    u_w, u_h = user_person.size

    # Bosh va iyak koordinatalarini aniqlash
    if head_box and len(head_box) == 4:
        ymin, xmin, ymax, xmax = [float(v) for v in head_box]
        head_top_y = (ymin / 1000.0) * u_h
        head_bottom_y = (ymax / 1000.0) * u_h
        head_left_x = (xmin / 1000.0) * u_w
        head_right_x = (xmax / 1000.0) * u_w
    else:
        # Standart taxmin
        head_top_y = u_h * 0.15
        head_bottom_y = u_h * 0.55
        head_left_x = u_w * 0.30
        head_right_x = u_w * 0.70

    head_h = max(30.0, head_bottom_y - head_top_y)
    head_cx = (head_left_x + head_right_x) / 2.0

    # Kostyum shablonidagi yoqa koordinatasi
    collar_target_y = suit_h * info.get("collar_y_ratio", 0.40)
    # Boshning o'lchami shablonda balandlikning taxminan 45-50% qismini egallasin
    target_head_h = suit_h * 0.32
    scale = target_head_h / head_h

    # Foydalanuvchi rasmini mos o'lchamda masshtablaymiz
    scaled_w = int(round(u_w * scale))
    scaled_h = int(round(u_h * scale))
    user_scaled = user_person.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    # Yangi koordinatalar
    scaled_head_cx = head_cx * scale
    scaled_chin_y = head_bottom_y * scale

    # Shablon markaziga joylashtirish
    # Chin (iyak) yoqa tepasida (collar_target_y atrofida) tushishi kerak
    suit_cx = suit_w / 2.0
    paste_x = int(round(suit_cx - scaled_head_cx))
    paste_y = int(round((collar_target_y - 20) - scaled_chin_y))

    # Oq fon yaratish (1086x1448)
    canvas = Image.new("RGBA", (suit_w, suit_h), (255, 255, 255, 255))

    # 1. Avval foydalanuvchini joylaymiz (boshi, bo'yni, tanasi)
    user_layer = Image.new("RGBA", (suit_w, suit_h), (0, 0, 0, 0))
    user_layer.paste(user_scaled, (paste_x, paste_y), user_scaled)

    # 2. Kostyumni foydalanuvchining ustiga joylaymiz (yoqa ochiq bo'lgani uchun bo'yin va bosh ko'rinib turadi)
    combined = Image.alpha_composite(user_layer, suit_img)

    # 3. Sof oq fonga tushiramiz
    final_canvas = Image.new("RGBA", (suit_w, suit_h), (255, 255, 255, 255))
    final_canvas.paste(combined, (0, 0), combined)

    # 4. 3x4 standart pikselga (354x472) keltirish
    photo_3x4 = final_canvas.convert("RGB").resize((PHOTO_WIDTH_PX, PHOTO_HEIGHT_PX), Image.Resampling.LANCZOS)
    return photo_3x4


def process_photo_with_suit(
    input_path: Path,
    output_dir: Path,
    head_box: Optional[list] = None,
    suit_id: str = "suit_blue"
) -> Dict[str, Any]:
    """
    To'liq jarayon:
    1. Rasmni o'qish
    2. Fonni ajratish (rembg orqali sof RGBA siluet olish)
    3. Tanlangan kostyumni kiygizish (overlay_suit)
    4. 3x4 JPG, 10x15 JPG (8 dona) va 10x15 PDF chop etish fayllarini yaratish
    """
    from image_processor import create_10x15_sheet, remove_background_and_make_white
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_img = Image.open(input_path)
    raw_img = ImageOps.exif_transpose(raw_img)

    # 1. Fonni toza oqartiramiz (OpenCV floodFill, 0.01 soniya)
    whitened = remove_background_and_make_white(raw_img)
    person_rgba = whitened.convert("RGBA")

    # 2. Kostyumni kiygizish
    photo_3x4 = overlay_suit(person_rgba, head_box=head_box, suit_id=suit_id)

    # 3. Yagona 3x4 rasm
    single_path = output_dir / f"photo_3x4_{suit_id}.jpg"
    photo_3x4.save(single_path, "JPEG", dpi=(DPI, DPI), quality=95)

    # 4. 10x15 sm varaq (8 ta)
    sheet_10x15 = create_10x15_sheet(photo_3x4)
    sheet_jpg_path = output_dir / f"sheet_10x15_{suit_id}.jpg"
    sheet_10x15.save(sheet_jpg_path, "JPEG", dpi=(DPI, DPI), quality=95)

    # 5. 10x15 PDF chop etish uchun (300 DPI)
    sheet_pdf_path = output_dir / f"sheet_10x15_{suit_id}.pdf"
    sheet_10x15.save(sheet_pdf_path, "PDF", resolution=DPI)

    suit_title = SUIT_CATALOG.get(suit_id, {}).get("title", "Kostyum")

    return {
        "single_3x4": single_path,
        "sheet_jpg": sheet_jpg_path,
        "sheet_pdf": sheet_pdf_path,
        "suit_id": suit_id,
        "suit_title": suit_title
    }
