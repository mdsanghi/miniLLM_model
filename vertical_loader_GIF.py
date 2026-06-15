from PIL import Image, ImageDraw, ImageFont, features
import os
import math

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

RAQM_SUPPORT = features.check("raqm")

W = 700
H = 720
SCALE = 3
SW = W * SCALE
SH = H * SCALE

blue = (10, 82, 212, 255)
text_color = (0, 0, 0, 255)
white_bg = (255, 255, 255, 255)
strip_color = (233, 243, 255, 255)

fps = 30
seconds = 8
frames_n = fps * seconds

language_sets = {
    "en": [
        "Internal Policy Check",
        "Income Verification",
        "Mala Credit Bureau",
        "DBR Assessment"
    ],
    "ar": [
        "التحقق من السياسات الداخلية",
        "التحقق من الدخل",
        "مكتب المعلومات الائتمانية",
        "تقييم نسبة عبء الدين"
    ]
}


def is_arabic(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def prepare_text(text):
    if is_arabic(text) and ARABIC_SUPPORT and not RAQM_SUPPORT:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text


font_paths = [
    "/Users/mayanksanghi/Library/Fonts/Almarai-Bold.ttf",
    "/Users/mayanksanghi/Library/Fonts/Almarai-Regular.ttf",
    "/Users/mayanksanghi/Library/Fonts/Almarai-ExtraBold.ttf",
    "/Users/mayanksanghi/Library/Fonts/Almarai-Light.ttf",
]

font_path = next((p for p in font_paths if os.path.exists(p)), None)

if font_path:
    if RAQM_SUPPORT:
        text_font = ImageFont.truetype(
            font_path,
            40,
            layout_engine=ImageFont.Layout.RAQM
        )
    else:
        text_font = ImageFont.truetype(font_path, 40)
else:
    text_font = ImageFont.load_default()


def create_loader(language, steps):
    gif_path = f"vertical_step_loader_{language}.gif"

    left_x = 100
    top_y = 110
    gap = 144

    circle_radius = 30
    line_width = 14

    if language == "ar":
        icon_x = W - 100
        text_anchor_x = W - 180
        text_anchor = "rm"
    else:
        icon_x = 100
        text_anchor_x = 190
        text_anchor = "lm"

    positions = [(icon_x, top_y + i * gap) for i in range(len(steps))]

    frames = []

    for frame_index in range(frames_n):
        t = frame_index / (frames_n - 1)
        completed_float = t * len(steps)

        img_big = Image.new("RGBA", (SW, SH), white_bg)
        draw = ImageDraw.Draw(img_big)

        for i, label in enumerate(steps):
            cx = positions[i][0] * SCALE
            cy = positions[i][1] * SCALE

            circle_r = circle_radius * SCALE
            line_w = line_width * SCALE

            if i < len(steps) - 1:
                next_y = positions[i + 1][1] * SCALE

                if completed_float > i + 1:
                    line_progress = 1
                elif completed_float > i:
                    line_progress = completed_float - i
                else:
                    line_progress = 0

                line_end_y = cy + (next_y - cy) * line_progress

                if line_progress > 0:
                    draw.line(
                        [(cx, cy + circle_r), (cx, line_end_y - circle_r)],
                        fill=blue,
                        width=line_w
                    )

            if completed_float >= i + 1:
                draw.ellipse(
                    [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                    fill=blue
                )

                tick_width = 10 * SCALE
                tick_points = [
                    (cx - 16 * SCALE, cy),
                    (cx - 4 * SCALE, cy + 14 * SCALE),
                    (cx + 20 * SCALE, cy - 14 * SCALE)
                ]

                draw.line(
                    tick_points,
                    fill=(255, 255, 255, 255),
                    width=tick_width,
                    joint="curve"
                )

            else:
                dots = 12
                dot_radius = 6 * SCALE
                spinner_radius = circle_r
                angle_offset = frame_index * 8

                for d in range(dots):
                    angle = math.radians((360 / dots) * d + angle_offset)
                    dx = cx + math.cos(angle) * spinner_radius
                    dy = cy + math.sin(angle) * spinner_radius

                    draw.ellipse(
                        [
                            dx - dot_radius,
                            dy - dot_radius,
                            dx + dot_radius,
                            dy + dot_radius
                        ],
                        fill=strip_color
                    )

        img = img_big.resize((W, H), Image.Resampling.LANCZOS)
        draw_final = ImageDraw.Draw(img)

        for i, label in enumerate(steps):
            _, cy = positions[i]
            final_text = prepare_text(label)
            text_options = {}

            if is_arabic(label) and RAQM_SUPPORT:
                text_options = {
                    "direction": "rtl",
                    "language": "ar"
                }

            draw_final.text(
                (text_anchor_x, cy),
                final_text,
                font=text_font,
                fill=text_color,
                anchor=text_anchor,
                **text_options
            )

        frames.append(img.convert("RGB"))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False
    )

    return gif_path


for language, steps in language_sets.items():
    gif_file = create_loader(language, steps)

    print(f"Created: {gif_file}")

if not ARABIC_SUPPORT:
    print("For correct Arabic rendering, install:")
    print("pip install arabic-reshaper python-bidi")
