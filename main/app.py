from flask import Flask, request, url_for, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import random

if 'VERCEL' in os.environ:
    import main.generate
else:
    import generate

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__)
if 'VERCEL' in os.environ:
    UPLOAD_FOLDER = '/tmp/uploads'
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CORS(app)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api', methods=['POST'])
def index():
    names = request.get_json()
    question = names.get('question') if names else None
    image_base64 = names.get('image') if names else None
    image_url = None
    filepath = None
    try:
        if image_base64:
            header, encoded = image_base64.split(',', 1) if ',' in image_base64 else (None, image_base64)
            image_data = base64.b64decode(encoded)
            filename = f'{random.randint(1,10**18)}.webp'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            image_url = url_for('uploaded_file', filename=filename, _external=True)
    except:
        image_url = None
    d = jsonify(main.generate.output(question, image_url))
    
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error deleting image file: {e}")
    return d

if __name__ == '__main__':
    app.run(debug=True)