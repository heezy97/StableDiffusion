from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image
import os
import torch
from transformers import CLIPTokenizer
import model_loader
import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 9  # min: 1, max: 14

## SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        text_prompt = request.form['text_prompt']
        # Comment out strength-related code
        # strength = float(request.form['strength'])
        generated_image = None

        # Check the datatype of text_prompt
        print(f"Type of text_prompt: {type(text_prompt)}")  # Verify type

        # Verify if text_prompt is a string
        if not isinstance(text_prompt, str):
            raise ValueError("Text prompt should be a string")

        print(f"Text Prompt: {text_prompt}")
        # print(f"Strength: {strength}")

        if 'input_image' in request.files:
            input_image_file = request.files['input_image']
            if input_image_file.filename != '':
                input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_image_file.filename)
                input_image_file.save(input_image_path)
                input_image = Image.open(input_image_path)
                print("Input image provided")
            else:
                input_image = None
                print("No input image provided")
        else:
            input_image = None

        print(f"Input Image: {input_image}")

        # Set a default strength value
        strength = 0.8

        output_image = pipeline.generate(
            prompt=text_prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,  # Keep the default strength value
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        print("After pipeline generate!!")

        # Remove the uploaded image file if it exists
        if input_image is not None:
            os.remove(input_image_path)
            print(f"Deleted uploaded image file: {input_image_path}")

        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        img_name = 'generated_image.jpg'
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        Image.fromarray(output_image).save(img_path)
        generated_image = img_name
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)})

    return jsonify({"generated_image": generated_image})

@app.route('/static/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
