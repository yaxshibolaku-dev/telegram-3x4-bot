import asyncio
import logging
import shutil
from pathlib import Path
import time

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, FSInputFile
from aiogram.client.default import DefaultBotProperties

import os
from aiohttp import web
import aiohttp

from config import BOT_TOKEN, TEMP_DIR, GEMINI_PRIMARY_MODEL
from image_processor import process_user_photo
from gemini_vision import analyze_photo_with_gemini, format_gemini_report

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


# --- Render Web Server va Keep-Alive Health Check ---
async def handle_ping(request):
    """Cron-job yoki UptimeRobot uchun oddiy keep-alive javob"""
    return web.Response(text="pong", status=200)


async def handle_health(request):
    """Render monitoringi va sog'lomlik tekshiruvi"""
    return web.json_response({
        "status": "online",
        "service": "3x4 Telegram Bot",
        "gemini_model": GEMINI_PRIMARY_MODEL,
        "timestamp": time.time()
    })


async def self_ping_loop(url: str):
    """Render bepul tarifida bot uxlab qolmasligi uchun o'zini har 10 daqiqada uyg'otib turadi."""
    await asyncio.sleep(45)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                ping_url = f"{url.rstrip('/')}/ping"
                async with session.get(ping_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    logger.info(f"Self keep-alive ping yuborildi: {resp.status}")
            except Exception as e:
                logger.warning(f"Self-ping xatosi: {e}")
            await asyncio.sleep(600)  # Har 10 daqiqada (600 soniya)


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ping", handle_ping)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Render Web Server muvaffaqiyatli ishga tushdi: 0.0.0.0:{port}")

    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if external_url:
        logger.info(f"Avtomatik keep-alive yoqildi: {external_url}")
        asyncio.create_task(self_ping_loop(external_url))


@dp.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "<b>Assalomu alaykum!</b> 📸\n\n"
        "Men hujjatlar uchun <b>3×4 fotosurat</b> va <b>10×15 sm bosma PDF</b> tayyorlab beruvchi aqlli botman.\n\n"
        "🤖 <b>Google Gemini Vision AI integratsiyasi:</b>\n"
        "▫️ Bosh tepasi, soch yoki iyak kesilib ketmaganligini tekshiradi.\n"
        "▫️ 3×4 o‘lcham va proporsiyalarni aniq nazorat qiladi.\n"
        "▫️ Yuz yo‘nalishi, ko‘zlar ochiqligi, yoritish va tiniqlikni aniqlab, sifat hisobotini taqdim etadi.\n\n"
        "<b>Bot qanday ishlaydi?</b>\n"
        "1. Gemini Vision AI rasmingizni ko‘rib, barcha xatolarni aniqlaydi.\n"
        "2. Fonni tozalab, sof oq (#FFFFFF) qiladi.\n"
        "3. Boshingiz kesilmasligini kafolatlagan holda 3×4 standartiga qirqadi.\n"
        "4. 10×15 sm fotolentaga 8 dona qilib (2×4 jadval) joylaydi.\n"
        "5. Sizga alohida 3×4 rasm, 8 talik JPG, to‘g‘ridan-to‘g‘ri printerga chiqariladigan <b>PDF</b> fayl va <b>AI tahlil xulosasini</b> yuboradi.\n\n"
        "<i>Menga o‘zingiz tushgan rasmni (Photo yoki Hujjat tarzida) yuboring!</i>"
    )
    await message.answer(text)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    text = (
        "<b>Qanday qilib mukammal 3×4 natija olish mumkin?</b>\n\n"
        "1. <b>Bosh holati:</b> Kameraga to‘g‘ri qarang, yuzingiz burilmagan bo‘lsin.\n"
        "2. <b>Kesilmaslik:</b> Rasm olayotganda bosh tepasi va sochingiz to‘liq kadrda bo‘lsin.\n"
        "3. <b>Yoritish:</b> Yuzingizga teng yorug‘lik tushgan bo‘lsin, kuchli soyalar bo‘lmasin.\n"
        "4. <b>Ko‘zlar:</b> Ko‘zlaringiz ochiq va to‘g‘riga qaratilgan bo‘lsin.\n"
        "5. <b>Orqa fon:</b> Fon muhim emas — bot fonni o‘zi oq qilib beradi.\n"
        "6. <b>Yuqori sifat:</b> Rasmni Telegram’da <b>'Fayl / Hujjat' (Document)</b> tarzida yuborish tavsiya etiladi.\n\n"
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

        # 3. Rasmga ishlov berish holati
        await status_msg.edit_text(
            "⚙️ <i>AI tahlili yakunlandi. Fon tozalanmoqda va bosh kesilmaydigan 3×4 formatga keltirilmoqda...</i>"
        )

        # 4. Rasmga ishlov berishni alohida thread'da bajarish
        results = await asyncio.to_thread(process_user_photo, input_path, user_temp_dir, head_box=head_box)

        # 5. Natijalarni yuborish
        # A) 3x4 bitta rasm
        single_file = FSInputFile(results["single_3x4"], filename="photo_3x4.jpg")
        await message.answer_photo(
            single_file,
            caption="✅ <b>3×4 sm Hujjat fotosurati (300 DPI)</b>\n<i>Sof oq fon (#FFFFFF) va standart proporsiya</i>"
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

        await status_msg.delete()

    except Exception as e:
        logger.error(f"Xatolik yuz berdi: {e}", exc_info=True)
        await status_msg.edit_text(
            "❌ <b>Kechirasiz, rasmni qayta ishlashda kutilmagan xatolik yuz berdi!</b>\n"
            "Iltimos, yuzingiz aniq ko‘ringan boshqa rasm yuborib ko‘ring."
        )
    finally:
        # Vaqtinchalik fayllarni o'chirish
        shutil.rmtree(user_temp_dir, ignore_errors=True)


@dp.message(F.photo)
async def on_photo_received(message: Message):
    # Eng yuqori sifatli rasmni tanlaymiz
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
    logger.info("Bot ishga tushmoqda...")
    # Render va monitoring uchun veb serverni ishga tushirish
    await start_web_server()
    # Eski kutilayotgan yangilanishlarni tozalash
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
