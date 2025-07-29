
from flask import Flask, render_template, request, send_file, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import io
import base64

app = Flask(__name__)


# Load the trained model
class DeepFaceDrawing(torch.nn.Module):
    def __init__(self):
        super(DeepFaceDrawing, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = DeepFaceDrawing()
model.load_state_dict(torch.load('deepfacedrawing_model.pkl', map_location=torch.device('cpu')))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Image upload and processing
@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded."})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected."})

    image = Image.open(file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()

    output_image = transforms.ToPILImage()(output_tensor * 0.5 + 0.5)
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)

    output_image_data = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return jsonify({"image": output_image_data})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
