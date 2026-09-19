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

# Kostyum va ko'ylak shablonlari katalogi (har birining yoqasiga mos kalibrovka qilingan)
SUIT_CATALOG = {
    "suit_blue": {
        "title": "To‘q ko‘k kostyum (galstukli)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_01 PM (1).png",
        "chin_y": 505,
        "crop_top": 130,
    },
    "suit_black_red": {
        "title": "Qora kostyum (qizil galstuk)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (2).png",
        "chin_y": 585,
        "crop_top": 210,
    },
    "suit_grey": {
        "title": "To‘q kulrang kostyum (ko‘k galstuk)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (3).png",
        "chin_y": 645,
        "crop_top": 270,
    },
    "suit_black": {
        "title": "Qora kostyum (galstuksiz)",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_39_02 PM (4).png",
        "chin_y": 695,
        "crop_top": 320,
    },
    "shirt_white": {
        "title": "Oq klassik ko‘ylak",
        "emoji": "👔",
        "file": "ChatGPT Image Sep 19, 2026, 01_38_55 PM (2).png",
        "chin_y": 645,
        "crop_top": 270,
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
    kostyum esa pastki kiyimlarni to'liq yopadi va 3x4 portret proporsiyasida qirqiladi.
    """
    suit_img, info = load_suit_template(suit_id)
    sw, sh = suit_img.size  # 1086 x 1448

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
        head_top_y = u_h * 0.15
        head_bottom_y = u_h * 0.55
        head_left_x = u_w * 0.30
        head_right_x = u_w * 0.70

    head_h = max(30.0, head_bottom_y - head_top_y)
    head_cx = (head_left_x + head_right_x) / 2.0

    # 3x4 shablonda bosh balandligi 320 px bo'lsin (standart proporsiya)
    target_head_h = 320.0
    scale = target_head_h / head_h

    scaled_w = int(round(u_w * scale))
    scaled_h = int(round(u_h * scale))
    user_scaled = user_person.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    scaled_head_cx = head_cx * scale
    scaled_chin_y = head_bottom_y * scale

    chin_target_y = info.get("chin_y", 580)
    crop_top = info.get("crop_top", 200)

    # Shablon markaziga joylashtirish (sw / 2 = 543)
    paste_x = int(round((sw / 2.0) - scaled_head_cx))
    paste_y = int(round(chin_target_y - scaled_chin_y))

    # Oq fon yaratish (1086x1448)
    canvas = Image.new("RGBA", (sw, sh), (255, 255, 255, 255))

    # 1. Foydalanuvchini joylaymiz (boshi va bo'yni)
    canvas.paste(user_scaled, (paste_x, paste_y), user_scaled)

    # 2. Kostyumni ustiga joylaymiz
    canvas.alpha_composite(suit_img)

    # 3. 3x4 portret formatda qirqish (bosh + yoqa + yelkalar, butun tanani emas!)
    crop_h = 533
    crop_w = int(round(crop_h * 0.75))  # 400 px
    crop_bottom = crop_top + crop_h
    crop_left = int(round((sw / 2.0) - (crop_w / 2.0)))
    crop_right = crop_left + crop_w

    cropped_suit = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 4. Standart 3x4 pikselga (354x472, 300 DPI) keltirish
    photo_3x4 = cropped_suit.convert("RGB").resize((PHOTO_WIDTH_PX, PHOTO_HEIGHT_PX), Image.Resampling.LANCZOS)
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
    2. U2NetP orqali fonni kesib, shaffof RGBA siluet olish
    3. Tanlangan kostyumni kiygizish (overlay_suit)
    4. 3x4 JPG, 10x15 JPG (8 dona) va 10x15 PDF chop etish fayllarini yaratish
    """
    from image_processor import create_10x15_sheet, get_person_mask
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_img = Image.open(input_path)
    raw_img = ImageOps.exif_transpose(raw_img)
    if raw_img.mode != "RGB":
        raw_img = raw_img.convert("RGB")

    # 1. Inson silueti maskasini olish va shaffof RGBA qilish
    mask_img = get_person_mask(raw_img)
    person_rgba = raw_img.convert("RGBA")
    person_rgba.putalpha(mask_img)

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
