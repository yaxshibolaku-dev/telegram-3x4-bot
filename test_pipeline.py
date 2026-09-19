from pathlib import Path
from PIL import Image, ImageDraw
import image_processor
import gemini_vision
from config import PHOTO_WIDTH_PX, PHOTO_HEIGHT_PX, SHEET_WIDTH_PX, SHEET_HEIGHT_PX


def create_sample_portrait(path: Path):
    # Rasm yaratish: 600x800, fon rangli
    img = Image.new("RGB", (600, 800), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Tana / Yelkalar (ellips)
    draw.ellipse([100, 450, 500, 900], fill=(50, 50, 80))
    
    # Bo'yin
    draw.rectangle([260, 400, 340, 500], fill=(230, 190, 160))
    
    # Bosh / Yuz (ellips)
    draw.ellipse([200, 200, 400, 450], fill=(240, 200, 170))
    
    # Ko'zlar
    draw.ellipse([250, 300, 275, 315], fill=(30, 30, 30))
    draw.ellipse([325, 300, 350, 315], fill=(30, 30, 30))
    
    # Burun
    draw.polygon([(300, 320), (290, 355), (310, 355)], fill=(210, 170, 140))
    
    # Og'iz
    draw.rectangle([280, 380, 320, 390], fill=(180, 70, 70))
    
    img.save(path, "JPEG")
    print(f"Sample image saved to {path}")


def create_non_person_image(path: Path):
    # Inson bo'lmagan rasm (tabiat manzarasi/geometrik shakl)
    img = Image.new("RGB", (400, 400), color=(30, 140, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 350], fill=(200, 200, 50))
    img.save(path, "JPEG")
    print(f"Non-person image saved to {path}")


def run_test():
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    sample_input = test_dir / "sample_portrait.jpg"
    create_sample_portrait(sample_input)

    non_person_input = test_dir / "non_person.jpg"
    create_non_person_image(non_person_input)
    
    print("\n--- 1. Gemini Vision AI tahlilini tekshirish ---")
    analysis = gemini_vision.analyze_photo_with_gemini_sync(sample_input)
    assert analysis.get("success"), f"Gemini tahlili muvaffaqiyatsiz bo'ldi: {analysis.get('error')}"
    print(f"Model ishlatildi: {analysis.get('model_used')}")
    print(f"Inson aniqlandimi: {analysis.get('has_person')}")
    print(f"Bosh kesilmaganmi: {not analysis.get('head_cut_off')}")
    print(f"Hujjatga moslik bali: {analysis.get('suitability_score')}/100")
    print(f"AI Head Box: {analysis.get('head_box_2d')}")

    report = gemini_vision.format_gemini_report(analysis)
    print(f"\nTelegram AI hisoboti:\n{report}\n")

    print("\n--- 2. Inson bo'lmagan rasm tekshiruvi ---")
    non_analysis = gemini_vision.analyze_photo_with_gemini_sync(non_person_input)
    print(f"Inson bormi deb topildi: {non_analysis.get('has_person')}")
    assert non_analysis.get("has_person") is False, "Inson bo'lmagan rasmda inson topilmasligi kerak!"

    print("\n--- 3. Gemini Head Box orqali to'liq ishlov berish ---")
    head_box = analysis.get("head_box_2d")
    results = image_processor.process_user_photo(sample_input, test_dir / "ai_output", head_box=head_box)
    
    single_img = Image.open(results["single_3x4"])
    print(f"Single 3x4 size: {single_img.size} (Expected: {PHOTO_WIDTH_PX}, {PHOTO_HEIGHT_PX})")
    assert single_img.size == (PHOTO_WIDTH_PX, PHOTO_HEIGHT_PX), "3x4 size mismatch!"
    
    sheet_img = Image.open(results["sheet_jpg"])
    print(f"10x15 Sheet size: {sheet_img.size} (Expected: {SHEET_WIDTH_PX}, {SHEET_HEIGHT_PX})")
    assert sheet_img.size == (SHEET_WIDTH_PX, SHEET_HEIGHT_PX), "10x15 sheet size mismatch!"
    
    pdf_path = results["sheet_pdf"]
    assert pdf_path.exists() and pdf_path.stat().st_size > 0, "PDF not created or empty!"
    print(f"PDF created successfully: {pdf_path.stat().st_size} bytes")
    
    print("\n--- 4. Kostyum va ko'ylak kiygizish tekshiruvi ---")
    import suit_processor
    suit_res = suit_processor.process_photo_with_suit(sample_input, test_dir / "suit_blue", head_box=head_box, suit_id="suit_blue")
    assert suit_res["single_3x4"].exists(), "Suit 3x4 not created!"
    assert suit_res["sheet_pdf"].exists(), "Suit PDF not created!"
    print(f"Suit 3x4 created: {suit_res['single_3x4']} ({Image.open(suit_res['single_3x4']).size})")

    shirt_res = suit_processor.process_photo_with_suit(sample_input, test_dir / "shirt_white", head_box=head_box, suit_id="shirt_white")
    assert shirt_res["single_3x4"].exists(), "Shirt 3x4 not created!"
    print(f"Shirt 3x4 created: {shirt_res['single_3x4']} ({Image.open(shirt_res['single_3x4']).size})")

    print("\n✅ BARCHA TESTLAR MUVAFFAQIYATLI O'TDI!")


if __name__ == "__main__":
    run_test()
