import os
import sys
import time
import json
import shutil
import logging
import asyncio
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request

print(">>> 3x4 Photo Telegram Bot jarayoni boshlanmoqda...", flush=True)

# -------------------------------------------------------------
# 1. Render Web Server (Port 10000 / 8080) darhol ochiladi
# -------------------------------------------------------------
PORT = int(os.getenv("PORT", "10000"))

class HealthHandler(BaseHTTPRequestHandler):
    def _respond(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        response = {
            "status": "online",
            "service": "3x4 Telegram Bot",
            "ping": "pong",
            "timestamp": time.time()
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self):
        self._respond()

    def do_HEAD(self):
        self._respond()

    def do_POST(self):
        self._respond()

    def log_message(self, format, *args):
        pass

def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    print(f">>> Render Web Server muvaffaqiyatli ishga tushdi: 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# -------------------------------------------------------------
# 2. Render Keep-Alive Loop (Bot uxlab qolmasligi uchun)
# -------------------------------------------------------------
def keep_alive_worker(external_url: str):
    time.sleep(30)
    ping_url = f"{external_url.rstrip('/')}/ping"
    print(f">>> Keep-alive monitoring faollashtirildi: {ping_url}", flush=True)
    while True:
        try:
            req = urllib.request.Request(ping_url, headers={"User-Agent": "RenderKeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f">>> Self-ping yuborildi: status {resp.status}", flush=True)
        except Exception as e:
            print(f">>> Self-ping ogohlantirish: {e}", flush=True)
        time.sleep(600)

EXT_URL = os.getenv("RENDER_EXTERNAL_URL")
if EXT_URL:
    threading.Thread(target=keep_alive_worker, args=(EXT_URL,), daemon=True).start()

# -------------------------------------------------------------
# 3. Asosiy Bot Modullari va AI Kutubxonalari
# -------------------------------------------------------------
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.default import DefaultBotProperties

from config import BOT_TOKEN, TEMP_DIR, GEMINI_PRIMARY_MODEL
from gemini_vision import analyze_photo_with_gemini, format_gemini_report
from suit_processor import SUIT_CATALOG, process_photo_with_suit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

SESSIONS_DIR = TEMP_DIR / "user_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_suit_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Kostyum va ko'ylaklarni tanlash uchun inline tugmalar"""
    buttons = [
        [
            InlineKeyboardButton(text="👔 To‘q ko‘k kostyum", callback_data=f"suit:suit_blue:{user_id}"),
            InlineKeyboardButton(text="👔 Qora (qizil galstuk)", callback_data=f"suit:suit_black_red:{user_id}")
        ],
        [
            InlineKeyboardButton(text="👔 Kulrang kostyum", callback_data=f"suit:suit_grey:{user_id}"),
            InlineKeyboardButton(text="👔 Qora (galstuksiz)", callback_data=f"suit:suit_black:{user_id}")
        ],
        [
            InlineKeyboardButton(text="👔 Oq klassik ko‘ylak", callback_data=f"suit:shirt_white:{user_id}"),
            InlineKeyboardButton(text="👤 O‘z kiyimida (asli)", callback_data=f"suit:original:{user_id}")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "<b>Assalomu alaykum!</b> 📸\n\n"
        "Men hujjatlar uchun <b>3×4 fotosurat</b>, <b>10×15 sm bosma PDF</b> tayyorlovchi va "
        "<b>virtual rasmiy kiyim (kostyum/ko‘ylak)</b> kiygizib beruvchi aqlli botman.\n\n"
        "🤖 <b>Imkoniyatlar:</b>\n"
        "▫️ <b>Gemini Vision AI:</b> Bosh kesilmaganligini, o‘lcham va sifatni tekshiradi.\n"
        "▫️ <b>Smart Crop:</b> Soch va bosh aslo kesilib ketmaydi.\n"
        "▫️ <b>Virtual Kostyum:</b> Rasmingizga klassik to‘q ko‘k, qora, kulrang kostyumlar yoki oq ko‘ylak kiygizing!\n"
        "▫️ <b>Chop etishga tayyor:</b> 10×15 sm fotolentaga 8 dona terilgan PDF fayl.\n\n"
        "<i>Menga o‘zingiz tushgan fotosuratni yuboring!</i>"
    )
    await message.answer(text)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    text = (
        "<b>Qanday qilib mukammal 3×4 natija olish mumkin?</b>\n\n"
        "1. <b>Bosh holati:</b> Kameraga to‘g‘ri qarang, yuzingiz burilmagan bo‘lsin.\n"
        "2. <b>Kesilmaslik:</b> Rasm olayotganda bosh tepasi va sochingiz to‘liq kadrda bo‘lsin.\n"
        "3. <b>Yoritish:</b> Yuzingizga teng yorug‘lik tushgan bo‘lsin.\n"
        "4. <b>Kostyum tanlash:</b> Rasm tayyor bo‘lgach, tugmalar orqali xohlagan kostyum yoki ko‘ylakni tanlashingiz mumkin.\n"
        "5. <b>Yuqori sifat:</b> Rasmni Telegram’da <b>'Fayl / Hujjat' (Document)</b> tarzida yuborish tavsiya etiladi.\n\n"
        "<i>Rasmingizni yuborib ko‘ring!</i>"
    )
    await message.answer(text)


async def handle_image_processing(message: Message, file_id: str, original_filename: str = "input.jpg"):
    user_id = message.from_user.id
    timestamp = int(time.time() * 1000)
    user_temp_dir = TEMP_DIR / f"{user_id}_{timestamp}"
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = user_temp_dir / original_filename

    status_msg = await message.reply(
        "⏳ <i>Rasmingiz qabul qilindi...</i>\n"
        "🤖 <b>Gemini Vision AI</b> <i>rasmni tahlil qilmoqda (bosh, o‘lcham, sifat tekshiruvi)...</i>"
    )

    try:
        # 1. Faylni yuklab olish
        telegram_file = await bot.get_file(file_id)
        await bot.download_file(telegram_file.file_path, destination=input_path)

        # 2. Gemini Vision AI orqali rasmni tekshirish
        analysis = await analyze_photo_with_gemini(input_path)

        # Agar sun'iy intellekt rasmda inson yuzini topolmasa
        if analysis.get("success") and analysis.get("has_person") is False:
            await status_msg.edit_text(
                "❌ <b>Rasmda inson yuzi aniqlanmadi!</b>\n\n"
                "💡 <i>Iltimos, inson yuzi va boshi aniq ko‘ringan haqiqiy fotosurat yuboring.</i>"
            )
            return

        head_box = analysis.get("head_box_2d") if analysis.get("success") else None

        # Foydalanuvchi seansini keyinchalik kostyum almashtirish uchun saqlaymiz
        user_sess_dir = SESSIONS_DIR / str(user_id)
        user_sess_dir.mkdir(parents=True, exist_ok=True)
        cached_input = user_sess_dir / "input.jpg"
        shutil.copyfile(input_path, cached_input)
        with open(user_sess_dir / "meta.json", "w", encoding="utf-8") as mf:
            json.dump({"head_box": head_box, "timestamp": time.time()}, mf)

        # 3. Rasmga ishlov berish holati
        await status_msg.edit_text(
            "⚙️ <i>AI tahlili yakunlandi. Fon tozalanmoqda va bosh kesilmaydigan 3×4 formatga keltirilmoqda...</i>"
        )

        # 4. Rasmga ishlov berishni alohida thread'da bajarish (maksimal 30s)
        from image_processor import process_user_photo
        results = await asyncio.wait_for(
            asyncio.to_thread(process_user_photo, input_path, user_temp_dir, head_box=head_box),
            timeout=30.0
        )

        # 5. Natijalarni yuborish
        # A) 3x4 bitta rasm
        single_file = FSInputFile(results["single_3x4"], filename="photo_3x4.jpg")
        await message.answer_photo(
            single_file,
            caption="✅ <b>3×4 sm Hujjat fotosurati (Asl kiyimida)</b>\n<i>Sof oq fon (#FFFFFF) va standart proporsiya</i>"
        )

        # B) 10x15 sm varaq (JPG ko'rinishida 8 ta rasm)
        sheet_jpg_file = FSInputFile(results["sheet_jpg"], filename="sheet_10x15.jpg")
        await message.answer_photo(
            sheet_jpg_file,
            caption="🖼 <b>10×15 sm Qog‘oz (8 dona 3×4 rasm)</b>\n<i>Kesish ramkalari bilan tayyorlangan</i>"
        )

        # C) 10x15 sm PDF chop etish uchun
        sheet_pdf_file = FSInputFile(results["sheet_pdf"], filename="3x4_pechat_10x15.pdf")
        await message.answer_document(
            sheet_pdf_file,
            caption=(
                "🖨 <b>Chop etish uchun tayyor PDF (10×15 sm, 300 DPI)</b>\n\n"
                "💡 <i>Ushbu PDF faylni foto-printerda 10×15 sm (4×6 dyuym) fotovaraqqa "
                "<b>100% (Actual size)</b> masshtabda chiqarsangiz, rasmlar aniq 3×4 sm o‘lchamda chiqadi.</i>"
            )
        )

        # D) Gemini Vision AI tahlil hisobotini yuborish
        report_text = format_gemini_report(analysis)
        await message.answer(report_text)

        # E) Kostyum/Ko'ylak tanlash taklifi
        keyboard = get_suit_keyboard(user_id)
        await message.answer(
            "👔 <b>Fotosuratingizga rasmiy kiyim kiygizishni xohlaysizmi?</b>\n\n"
            "Quyidagi tugmalardan birini tanlab, fotosuratingizga klassik kostyum yoki oq ko‘ylak kiygizishingiz mumkin:",
            reply_markup=keyboard
        )

        await status_msg.delete()

    except Exception as e:
        logger.error(f"Xatolik yuz berdi: {e}", exc_info=True)
        await status_msg.edit_text(
            "❌ <b>Kechirasiz, rasmni qayta ishlashda kutilmagan xatolik yuz berdi!</b>\n"
            "Iltimos, yuzingiz aniq ko‘ringan boshqa rasm yuborib ko‘ring."
        )
    finally:
        shutil.rmtree(user_temp_dir, ignore_errors=True)


@dp.callback_query(F.data.startswith("suit:"))
async def on_suit_selected(callback: CallbackQuery):
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Noto‘g‘ri so‘rov")
        return

    suit_id = parts[1]
    user_id = int(parts[2])

    # Faqat o'z rasmini boshqara olsin
    if callback.from_user.id != user_id:
        await callback.answer("Bu tugma faqat rasm egasi uchun!", show_alert=True)
        return

    user_sess_dir = SESSIONS_DIR / str(user_id)
    cached_input = user_sess_dir / "input.jpg"
    meta_file = user_sess_dir / "meta.json"

    if not cached_input.exists():
        await callback.answer("Seans muddati tugagan. Iltimos, rasmni qayta yuboring.", show_alert=True)
        return

    head_box = None
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                head_box = meta.get("head_box")
        except Exception:
            pass

    await callback.answer("Kiyim moslashtirilmoqda...")
    status_msg = await callback.message.reply("⏳ <i>Tanlangan kiyim moslashtirilmoqda va 3×4 tayyorlanmoqda... Iltimos, kuting.</i>")

    timestamp = int(time.time() * 1000)
    out_dir = TEMP_DIR / f"suit_{user_id}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if suit_id == "original":
            from image_processor import process_user_photo
            results = await asyncio.wait_for(
                asyncio.to_thread(process_user_photo, cached_input, out_dir, head_box=head_box),
                timeout=30.0
            )
            title = "O‘z kiyimida (asli)"
        else:
            results = await asyncio.wait_for(
                asyncio.to_thread(process_photo_with_suit, cached_input, out_dir, head_box=head_box, suit_id=suit_id),
                timeout=30.0
            )
            title = results.get("suit_title", "Klassik kiyim")

        # Natijalarni yuborish
        single_file = FSInputFile(results["single_3x4"], filename=f"photo_3x4_{suit_id}.jpg")
        await callback.message.answer_photo(
            single_file,
            caption=f"✅ <b>3×4 sm Hujjat fotosurati ({title})</b>\n<i>Sof oq fon (#FFFFFF) va standart proporsiya</i>"
        )

        sheet_jpg_file = FSInputFile(results["sheet_jpg"], filename=f"sheet_10x15_{suit_id}.jpg")
        await callback.message.answer_photo(
            sheet_jpg_file,
            caption=f"🖼 <b>10×15 sm Qog‘oz (8 dona 3×4 rasm - {title})</b>"
        )

        sheet_pdf_file = FSInputFile(results["sheet_pdf"], filename=f"3x4_{suit_id}_10x15.pdf")
        await callback.message.answer_document(
            sheet_pdf_file,
            caption=(
                f"🖨 <b>Chop etish uchun tayyor PDF ({title}, 10×15 sm, 300 DPI)</b>\n\n"
                "💡 <i>Foto-printerda 10×15 sm (4×6 dyuym) fotovaraqqa 100% masshtabda chop eting.</i>"
            )
        )

        keyboard = get_suit_keyboard(user_id)
        await callback.message.answer(
            "💡 <i>Boshqa kiyim uslubini ham sinab ko‘rishingiz mumkin:</i>",
            reply_markup=keyboard
        )

        await status_msg.delete()

    except Exception as e:
        logger.error(f"Kostyum o'rnatishda xatolik: {e}", exc_info=True)
        await status_msg.edit_text("❌ <i>Kiyimni moslashda xatolik yuz berdi. Iltimos, boshqa rasm bilan urinib ko‘ring.</i>")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


@dp.message(F.photo)
async def on_photo_received(message: Message):
    highest_photo = message.photo[-1]
    await handle_image_processing(message, highest_photo.file_id, "input.jpg")


@dp.message(F.document)
async def on_document_received(message: Message):
    doc = message.document
    mime = (doc.mime_type or "").lower()
    filename = (doc.file_name or "").lower()

    if mime.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        ext = Path(filename).suffix or ".jpg"
        await handle_image_processing(message, doc.file_id, f"input{ext}")
    else:
        await message.reply("Iltimos, rasm faylini yuboring (JPG, PNG, WEBP).")


async def main():
    print(">>> --------------------------------------------------", flush=True)
    print(">>> Telegram Bot polling boshlanmoqda...", flush=True)
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        print(">>> Webhook tekshirildi va tozalandi.", flush=True)
        me = await bot.get_me()
        print(f">>> Telegram Bot muvaffaqiyatli ulandi: @{me.username} (ID: {me.id})", flush=True)
        print(">>> dp.start_polling boshlanmoqda (message, callback_query)...", flush=True)
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    except Exception as e:
        print(f">>> CRITICAL ERROR in bot polling: {e}", flush=True)
        logger.exception("Polling critical failure")


if __name__ == "__main__":
    asyncio.run(main())
