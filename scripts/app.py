from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import uuid

from better_polygonize_offset import polygonize_offset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

ALLOWED_JSON = {"json"}
ALLOWED_IMG = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        json_file = request.files.get("json_file")
        img_file = request.files.get("img_file")

        if not json_file or not img_file:
            return render_template("index.html", result=None)

        if not allowed_file(json_file.filename, ALLOWED_JSON):
            return "JSON file required", 400
        if not allowed_file(img_file.filename, ALLOWED_IMG):
            return "Image file required", 400

        # tao id duy nhat cho phien xu ly
        uid = uuid.uuid4().hex

        json_name = secure_filename(f"{uid}_{json_file.filename}")
        img_name = secure_filename(f"{uid}_{img_file.filename}")

        json_path = os.path.join(app.config["UPLOAD_FOLDER"], json_name)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_name)
        json_file.save(json_path)
        img_file.save(img_path)

        # output base path (khong uoi)
        output_base = os.path.join(app.config["OUTPUT_FOLDER"], uid)

        fill_path, line_path, orig_path, hybrid_svg, line_svg, orig_svg = \
            polygonize_offset(json_path, img_path, output_base)


        hybrid_path = output_base + '_hybrid.svg'

        result = {
            'fill': os.path.basename(hybrid_svg),
            'line': os.path.basename(line_svg),
            'orig': os.path.basename(orig_svg),
            'hybrid': os.path.basename(hybrid_path)  # Them download hybrid
        }

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

@app.route("/download/<path:filename>")
def download_file(filename):
    # static/outputs/<filename>
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, mimetype="image/png", as_attachment=True)

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8008
    )

