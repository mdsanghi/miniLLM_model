from PIL import Image, ImageDraw, ImageFont
import zipfile
import os

gif_path = "loading_indicator_white_bg.gif"
zip_path = "loading_indicator_white_bg.zip"
png_path = "loading_indicator_100_percent_white_bg.png"

W = H = 500
SCALE = 4
SW = SH = W * SCALE

blue = (10, 82, 212, 255)           # #0a52d4
strip_color = (233, 243, 255, 255)  # #E9F3FF
white_bg = (255, 255, 255, 255)

fps = 30
seconds = 12
frames_n = fps * seconds

font_paths = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

font_path = next((p for p in font_paths if os.path.exists(p)), None)
font = ImageFont.truetype(font_path, 70 * SCALE) if font_path else ImageFont.load_default()

cx = SW // 2
cy = SH // 2
radius = 170 * SCALE
stroke = 20 * SCALE

frames = []

def create_progress_image(percent):
    img = Image.new("RGBA", (SW, SH), white_bg)
    draw = ImageDraw.Draw(img)

    start_angle = -90
    end_angle = start_angle + (percent / 100) * 360

    box = [
        cx - radius,
        cy - radius,
        cx + radius,
        cy + radius
    ]

    draw.arc(
        box,
        0,
        360,
        fill=strip_color,
        width=stroke
    )

    if percent > 0:
        draw.arc(
            box,
            start_angle,
            end_angle,
            fill=blue,
            width=stroke
        )

    text = f"{percent}%"
    text_box = draw.textbbox((0, 0), text, font=font)

    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    draw.text(
        (
            cx - text_width / 2,
            cy - text_height / 2
        ),
        text,
        font=font,
        fill=blue
    )

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    return img.convert("RGB")


for i in range(frames_n):
    t = i / (frames_n - 1)
    percent = int(round(98 * t))
    frames.append(create_progress_image(percent))

png_image = create_progress_image(100)
png_image.save(png_path)

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / fps),
    loop=0,
    optimize=False
)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(gif_path, arcname=os.path.basename(gif_path))
    z.write(png_path, arcname=os.path.basename(png_path))

print("GIF created:", gif_path)
print("PNG created:", png_path)
print("ZIP created:", zip_path)
