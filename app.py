from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import os
from PIL import Image
import io
from generator import StringArtGenerator

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/generate', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.decodebytes(image_data.encode())
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    if not os.path.exists('static'): os.makedirs('static')
    temp_path = os.path.join('static', 'temp_input.png')
    img.save(temp_path)

    gen = StringArtGenerator(temp_path, num_pins=int(data['pins']))
    result_data = gen.generate(lines_to_draw=int(data['lines']))

    return jsonify({
        "result_url": "/download/string_art_result.png",
        "instruction_url": "/download/instrukcja.txt",
        "sequence": result_data["sequence"],
        "pins": result_data["pins"]
    })


if __name__ == '__main__':
    app.run(debug=True)