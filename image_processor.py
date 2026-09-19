import io
import time
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from config import (
    PHOTO_WIDTH_PX,
    PHOTO_HEIGHT_PX,
    SHEET_WIDTH_PX,
    SHEET_HEIGHT_PX,
    GRID_COLS,
    GRID_ROWS,
    DPI
)

_face_cascade = None
_onnx_session = None


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade


def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        import onnxruntime as ort
        from config import BASE_DIR
        model_path = BASE_DIR / "models" / "u2netp.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model fayli topilmadi: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    return _onnx_session


def get_person_mask(image: Image.Image) -> Image.Image:
    """
    Rasm ichidagi inson siluetini U2NetP modeli orqali 0.3 soniyada aniqlaydi va
    oq-qora (255=inson, 0=fon) niqob (mask) qaytaradi.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    try:
        session = get_onnx_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        resized = image.resize((320, 320), Image.Resampling.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = np.transpose(arr, (2, 0, 1))
        input_tensor = np.expand_dims(arr, axis=0).astype(np.float32)

        outputs = session.run([output_name], {input_name: input_tensor})
        mask = outputs[0][0, 0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask_uint8).resize((w, h), Image.Resampling.BILINEAR)
    except Exception as e:
        print(f">>> ONNX mask xatolik: {e}", flush=True)
        return Image.new("L", (w, h), 255)


def remove_background_and_make_white(image: Image.Image) -> Image.Image:
    """
    Rasm fonini toza oq (#FFFFFF) rangga aylantiradi.
    Lokal u2netp.onnx neyrotarmog'i orqali inson yuzi va sochlarini saqlagan holda
    0.4 soniyada fonni kesadi.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        mask_img = get_person_mask(image)
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        return Image.composite(image, white_bg, mask_img)
    except Exception as e:
        print(f">>> Fonni oqartirishda xatolik: {e}", flush=True)
        return image


def detect_and_crop_3x4(image: Image.Image, head_box: list = None) -> Image.Image:
    """
    Yuz va boshni aniqlab, 3x4 proporsiyasida (bosh tepasidan yetarli bo'sh joy qoldirib,
    boshi kesilib ketmasligini kafolatlagan holda) markazlashtirib qirqadi.
    Agar Gemini Vision tomonidan head_box [ymin, xmin, ymax, xmax] (0-1000 shkalada) berilgan bo'lsa,
    sun'iy intellekt koordinatalariga tayanadi.
    Aks holda OpenCV Haar cascade zaxira mexanizmidan foydalanadi.
    """
    img_w, img_h = image.size
    target_ratio = 3.0 / 4.0  # 0.75

    crop_left = 0
    crop_top = 0
    crop_right = img_w
    crop_bottom = img_h
    detected = False

    # 1. Agar Gemini Vision tomonidan bosh koordinatalari berilgan bo'lsa
    if head_box and isinstance(head_box, (list, tuple)) and len(head_box) == 4:
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in head_box]
            # 0-1000 shkalasini piksel koordinatalariga aylantirish
            top_head = (ymin / 1000.0) * img_h
            left_head = (xmin / 1000.0) * img_w
            bottom_chin = (ymax / 1000.0) * img_h
            right_head = (xmax / 1000.0) * img_w

            head_h = max(30.0, bottom_chin - top_head)
            head_cx = (left_head + right_head) / 2.0

            # 3x4 hujjat standarti:
            # Bosh balandligi (soch tepasidan iyak tagigacha) rasm umumiy balandligining ~58-62% qismini egallasin
            crop_h = head_h / 0.60
            crop_w = crop_h * target_ratio

            # Bosh tepasidan (sochdan yuqori) 12% bo'sh joy qoldiramiz (headroom)
            # Bu soch/bosh hech qachon kesilmasligini kafolatlaydi!
            crop_top = top_head - (crop_h * 0.12)
            crop_bottom = crop_top + crop_h
            crop_left = head_cx - (crop_w / 2.0)
            crop_right = crop_left + crop_w
            detected = True
        except Exception as e:
            detected = False

    # 2. Agar Gemini koordinatalari bo'lmasa, OpenCV Haar cascade bilan aniqlaymiz
    if not detected:
        try:
            import cv2
            cascade = get_face_cascade()
            np_img = np.array(image)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
        except Exception as e:
            faces = []

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            # Yuz balandligiga qarab 3x4 o'lcham hisoblash
            crop_h = int(fh / 0.52)
            crop_w = int(round(crop_h * target_ratio))

            face_cx = fx + fw // 2
            # Bosh tepasidan kamida 38% joy qoldiramiz (soch kesilmasligi uchun)
            crop_top = fy - int(fh * 0.38)
            crop_left = face_cx - crop_w // 2
            crop_bottom = crop_top + crop_h
            crop_right = crop_left + crop_w
            detected = True
        else:
            # Yuz umuman topilmasa, markaziy 3:4 qirqish
            current_ratio = img_w / img_h
            if current_ratio > target_ratio:
                new_w = int(round(img_h * target_ratio))
                crop_left = (img_w - new_w) // 2
                crop_right = crop_left + new_w
                crop_top = 0
                crop_bottom = img_h
            else:
                new_h = int(round(img_w / target_ratio))
                crop_top = int((img_h - new_h) * 0.15)
                crop_top = max(0, min(crop_top, img_h - new_h))
                crop_bottom = crop_top + new_h
                crop_left = 0
                crop_right = img_w

    crop_left = int(round(crop_left))
    crop_top = int(round(crop_top))
    crop_right = int(round(crop_right))
    crop_bottom = int(round(crop_bottom))

    # Agar qirqish chegaralari asl rasm chegarasidan chiqsa, oq fonga kengaytiramiz (pad)
    # Shunda bosh tepasi yoki yelkalar chetga taqalsa ham kesilib ketmaydi!
    pad_left = max(0, -crop_left)
    pad_top = max(0, -crop_top)
    pad_right = max(0, crop_right - img_w)
    pad_bottom = max(0, crop_bottom - img_h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_w = img_w + pad_left + pad_right
        padded_h = img_h + pad_top + pad_bottom
        padded_img = Image.new("RGB", (padded_w, padded_h), (255, 255, 255))
        padded_img.paste(image, (pad_left, pad_top))

        crop_left += pad_left
        crop_top += pad_top
        crop_right += pad_left
        crop_bottom += pad_top
        cropped = padded_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    else:
        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Standart 3x4 o'lchamga keltirish (354x472 px)
    photo_3x4 = cropped.resize((PHOTO_WIDTH_PX, PHOTO_HEIGHT_PX), Image.Resampling.LANCZOS)
    return photo_3x4



def add_border(image: Image.Image, border_width: int = 1, color: tuple = (0, 0, 0)) -> Image.Image:
    """
    Rasm chetiga qaychi bilan qirqish oson bo'lishi uchun ingichka qora ramka chizadi.
    """
    bordered = image.copy()
    draw = ImageDraw.Draw(bordered)
    w, h = bordered.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return bordered


def create_10x15_sheet(photo_3x4: Image.Image) -> Image.Image:
    """
    10x15 sm (1772x1181 px) o'lchamli varaqqa 8 ta 3x4 rasm joylashtiradi (4 ustun x 2 qator).
    """
    photo_with_border = add_border(photo_3x4, border_width=1, color=(30, 30, 30))
    sheet = Image.new("RGB", (SHEET_WIDTH_PX, SHEET_HEIGHT_PX), (255, 255, 255))

    # Oradagi masofalar (probellar)
    gap_x = 30
    gap_y = 30

    total_content_w = (GRID_COLS * PHOTO_WIDTH_PX) + ((GRID_COLS - 1) * gap_x)
    total_content_h = (GRID_ROWS * PHOTO_HEIGHT_PX) + ((GRID_ROWS - 1) * gap_y)

    margin_x = (SHEET_WIDTH_PX - total_content_w) // 2
    margin_y = (SHEET_HEIGHT_PX - total_content_h) // 2

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = margin_x + col * (PHOTO_WIDTH_PX + gap_x)
            y = margin_y + row * (PHOTO_HEIGHT_PX + gap_y)
            sheet.paste(photo_with_border, (x, y))

    return sheet


def process_user_photo(input_path: Path, output_dir: Path, head_box: list = None) -> dict:
    """
    To'liq jarayon:
    1. Rasmni o'qish
    2. Fonni oq qilish
    3. 3x4 ga bosh va yuzni moslab (bosh kesilmasligini kafolatlab) qirqish
    4. Alohida 3x4 rasm saqlash
    5. 10x15 sm 8 talik varaq yasash
    6. JPG va PDF formatlarda saqlash
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_img = Image.open(input_path)

    # 1. EXIF orientatsiyasini to'g'rilash
    clean_img = remove_background_and_make_white(raw_img)

    # 2. Bosh/yuzni aniqlash va 3x4 ga qirqish (Gemini head_box yordamida)
    photo_3x4 = detect_and_crop_3x4(clean_img, head_box=head_box)

    # 3. Yagona 3x4 rasmni saqlash (chegarasiz va chegarali)
    single_path = output_dir / "photo_3x4.jpg"
    photo_3x4.save(single_path, "JPEG", dpi=(DPI, DPI), quality=95)

    # 4. 10x15 sm varaq yasash
    sheet_10x15 = create_10x15_sheet(photo_3x4)

    # 5. 10x15 JPG saqlash
    sheet_jpg_path = output_dir / "sheet_10x15.jpg"
    sheet_10x15.save(sheet_jpg_path, "JPEG", dpi=(DPI, DPI), quality=95)

    # 6. 10x15 PDF saqlash (chop etish uchun 300 DPI)
    sheet_pdf_path = output_dir / "sheet_10x15.pdf"
    sheet_10x15.save(sheet_pdf_path, "PDF", resolution=DPI)

    return {
        "single_3x4": single_path,
        "sheet_jpg": sheet_jpg_path,
        "sheet_pdf": sheet_pdf_path
    }
