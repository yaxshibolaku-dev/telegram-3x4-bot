import io
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp
import requests
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

ANALYSIS_PROMPT = """
Siz 3x4 hujjat fotosuratlari (pasport, guvohnoma, talabalik, viza) bo'yicha professional ekspert sun'iy intellektsiz.
Ushbu rasm 3x4 hujjat fotosurati talablariga mosligini qat'iy va sinchkovlik bilan tahlil qiling:

1. Rasmda inson yuzi mavjudmi?
2. Bir kishimi yoki ko'pchilikmi?
3. Boshning tepasi (soch/peshona), chetlari yoki iyagi rasm hoshiyasida KESILIB KETMAGANMI?
4. Yuz kameraga to'g'ri qaraganmi (frontal) yoki yon tomonga burilganmi?
5. Ko'zlar ochiqmi va kameraga qaraganmi?
6. Yuz tiniqligi (sharp, acceptable yoki blurry)?
7. Yoritish sifati (yetarlimi, qorong'imi, soya bormi yoki haddan tashqari yorug'mi)?
8. 3x4 o'lcham uchun boshning [ymin, xmin, ymax, xmax] 0-1000 shkalasidagi to'liq koordinatalari (bosh tepasidan iyak tagigacha).

Javobni FAQAT quyidagi JSON formatida qaytaring:
{
  "has_person": true,
  "single_person": true,
  "head_cut_off": false,
  "head_cut_off_details": "Bosh to'liq ko'ringan, soch yoki iyak kesilmagan.",
  "facing_forward": true,
  "eyes_open": true,
  "face_clarity": "sharp",
  "lighting": "good",
  "suitability_score": 95,
  "issues": [],
  "recommendations": ["Rasm 3x4 hujjat talablariga to'liq javob beradi."],
  "head_box_2d": [200, 300, 550, 700]
}

Izohlar:
- head_cut_off: Agar rasmning asl nusxasida soch tepasi, yuzning yon qismi yoki iyak kesilgan bo'lsa true, aks holda false.
- head_box_2d: Boshning eng yuqori nuqtasidan (soch uchi) to iyakning pastki nuqtasigacha bo'lgan 4 ta koordinata [ymin, xmin, ymax, xmax] (0 dan 1000 gacha sonlar).
- issues va recommendations o'zbek tilida aniq, tushunarli bo'lsin.
"""


def _prepare_image_base64(image_input: Any, max_dim: int = 1200) -> str:
    """
    Rasmni o'qiydi, optimal hajmga keltiradi va Base64 formatga o'tkazadi.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input.copy()
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise ValueError("Noto'g'ri rasm formati")

    # RGB ga o'tkazish
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Juda katta rasmlarni tarmoqda tez uzatish uchun proporsional kichraytirish
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def analyze_photo_with_gemini(image_input: Any) -> Dict[str, Any]:
    """
    Rasmni Gemini Vision AI orqali asinxron tahlil qiladi.
    Bosh kesilmaganligi, yuz holati, o'lcham va xatolarni tekshiradi.
    gemini-3.5-flash-lite -> gemini-3.1-flash-lite modellaridan ketma-ket foydalanadi.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY mavjud emas! AI tahlil o'tkazib yuboriladi.")
        return {"success": False, "error": "API kalit sozlanmagan"}

    try:
        img_b64 = _prepare_image_base64(image_input)
    except Exception as e:
        logger.error(f"Rasmni tayyorlashda xatolik: {e}")
        return {"success": False, "error": f"Rasmni o'qib bo'lmadi: {e}"}

    payload = {
        "contents": [{
            "parts": [
                {"text": ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    # Sinab ko'riladigan modellar ro'yxati (asosiy va zaxiralar)
    models_to_try = [
        GEMINI_PRIMARY_MODEL,
        GEMINI_FALLBACK_MODEL,
        "gemini-2.5-flash-lite"
    ]
    # Takrorlanmas qilib tartiblash
    models_to_try = list(dict.fromkeys(filter(None, models_to_try)))

    timeout = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for model in models_to_try:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={GEMINI_API_KEY}"
            logger.info(f"Gemini Vision tahlili boshlanmoqda (Model: {model})...")

            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                raw_json = parts[0]["text"]
                                parsed = json.loads(raw_json)
                                parsed["success"] = True
                                parsed["model_used"] = model
                                logger.info(f"Gemini tahlili muvaffaqiyatli yakunlandi ({model})")
                                return parsed
                    else:
                        error_text = await resp.text()
                        logger.warning(f"Model {model} xatolik berdi: {resp.status} - {error_text[:200]}")
            except asyncio.TimeoutError:
                logger.warning(f"Model {model} bo'yicha timeout (20s) yuz berdi. Keyingi modelga o'tilmoqda.")
            except Exception as e:
                logger.warning(f"Model {model} bilan so'rovda xatolik: {e}")

    logger.error("Barcha Gemini modellarida xatolik yuz berdi!")
    return {"success": False, "error": "Gemini AI serveri bilan ulanishda xatolik yuz berdi"}


def analyze_photo_with_gemini_sync(image_input: Any) -> Dict[str, Any]:
    """
    Sinxron muhitlar yoki testlar uchun Gemini Vision tahlil funksiyasi.
    """
    if not GEMINI_API_KEY:
        return {"success": False, "error": "API kalit sozlanmagan"}

    img_b64 = _prepare_image_base64(image_input)
    payload = {
        "contents": [{
            "parts": [
                {"text": ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    models_to_try = [
        GEMINI_PRIMARY_MODEL,
        GEMINI_FALLBACK_MODEL,
        "gemini-2.5-flash-lite"
    ]
    models_to_try = list(dict.fromkeys(filter(None, models_to_try)))

    for model in models_to_try:
        url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                raw_json = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(raw_json)
                parsed["success"] = True
                parsed["model_used"] = model
                return parsed
        except Exception as e:
            logger.warning(f"Sinxron tahlilda model {model} xatolik: {e}")

    return {"success": False, "error": "Sinxron AI tahlil amalga oshmadi"}


def format_gemini_report(analysis: Dict[str, Any]) -> str:
    """
    Gemini tahlili natijalarini foydalanuvchiga chiroyli va tushunarli
    Telegram HTML xabari ko'rinishida formatlaydi.
    """
    if not analysis or not analysis.get("success"):
        return (
            "⚠️ <b>Sun'iy intellekt xulosasi:</b>\n"
            "<i>AI tahlili vaqtinchalik mavjud emas, biroq rasmingiz standart avtomatlashtirilgan usulda tayyorlandi.</i>"
        )

    score = analysis.get("suitability_score", 85)
    if score >= 85:
        badge = "🟢 <b>A'lo darajada mos</b>"
    elif score >= 65:
        badge = "🟡 <b>Qoniqarli</b>"
    else:
        badge = "🔴 <b>Diqqat talab qilinadi</b>"

    # 1. Bosh kesilmaganligi
    head_cut = analysis.get("head_cut_off", False)
    if head_cut:
        head_status = "⚠️ <b>Bosh qismi chetdan kesilgan</b>"
        head_detail = f"   └ <i>{analysis.get('head_cut_off_details', 'Bosh yoki soch qismi chetga taqalgan')}</i>\n"
    else:
        head_status = "✅ <b>To‘liq (boshi kesilmagan, joylashuv to‘g‘ri)</b>"
        head_detail = ""

    # 2. Yuz yo'nalishi va ko'zlar
    facing = analysis.get("facing_forward", True)
    eyes = analysis.get("eyes_open", True)
    if facing and eyes:
        face_status = "✅ To‘g‘riga qaragan, ko‘zlar ochiq"
    elif not facing and eyes:
        face_status = "⚠️ Yuz biroz chetga burilgan (frontal emas)"
    elif facing and not eyes:
        face_status = "⚠️ Ko‘zlar yumuq yoki qisiq ko‘ringan"
    else:
        face_status = "⚠️ Yuz burilgan va ko‘zlar to‘liq ko‘rinmagan"

    # 3. Yoritish va ravshanlik
    clarity_map = {
        "sharp": "Yuqori tiniqlik",
        "acceptable": "Yetarli ravshanlik",
        "blurry": "Biroz xira / noaniq"
    }
    clarity = clarity_map.get(analysis.get("face_clarity"), "Yetarli")
    
    lighting_map = {
        "good": "Yorug'lik to'g'ri taqsimlangan",
        "dark": "Biroz qorong'i",
        "overexposed": "Haddan tashqari yorug'",
        "shadowed": "Yuzda kuchli soya bor"
    }
    lighting = lighting_map.get(analysis.get("lighting"), "Yaxshi")
    quality_str = f"{clarity}, {lighting}"

    # 4. 3x4 O'lcham va proporsiya
    prop_status = "✅ 3×4 sm standart proporsiyada markazlashtirildi"

    # Muammolar va tavsiyalar
    issues = analysis.get("issues", [])
    recommendations = analysis.get("recommendations", [])

    lines = [
        "🤖 <b>Gemini Vision AI Tahlil Xulosasi:</b>\n",
        f"📊 <b>Hujjatga moslik darajasi:</b> {score}% ({badge})",
        f"👤 <b>Bosh holati:</b> {head_status}",
    ]
    if head_detail:
        lines.append(head_detail)

    lines.extend([
        f"👀 <b>Nigoh va ifoda:</b> {face_status}",
        f"💡 <b>Tiniqlik va yoritish:</b> {quality_str}",
        f"📐 <b>3×4 O‘lcham va ramka:</b> {prop_status}"
    ])

    if issues:
        lines.append("\n⚠️ <b>Aniqlangan kamchiliklar:</b>")
        for iss in issues:
            lines.append(f"• <i>{iss}</i>")

    if recommendations and not (len(recommendations) == 1 and "to'liq javob" in recommendations[0].lower()):
        lines.append("\n💡 <b>Tavsiya:</b>")
        for rec in recommendations:
            lines.append(f"• <i>{rec}</i>")

    model_name = analysis.get("model_used", GEMINI_PRIMARY_MODEL)
    lines.append(f"\n⚡ <i>Tahlilchi: Google {model_name}</i>")

    return "\n".join(lines)
