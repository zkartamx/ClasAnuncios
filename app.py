import dspy
import json
import os
import requests # Importar requests para la prueba de conexión
from flask import Flask, render_template, request, redirect, url_for, jsonify
import openai # Importar la librería de OpenAI
import google.generativeai as genai # Importar la librería de Google Generative AI

# --- Configuración de la Aplicación Flask ---
app = Flask(__name__)

# --- Funciones para manejar la configuración del modelo ---
CONFIG_FILE = 'config.json'

def load_config():
    if not os.path.exists(CONFIG_FILE):
        # Configuración por defecto: usar LM Studio
        initial_config = {
            'model_type': 'lm_studio',
            'lm_studio_ip': '192.168.100.49', # IP por defecto
            'lm_studio_port': '1234', # Puerto por defecto
            'openai_api_key': '', # Vacío por defecto
            'google_api_key': '' # Vacío por defecto
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(initial_config, f, ensure_ascii=False, indent=4)
        print(f"Created initial {CONFIG_FILE}")

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
        # Asegurarse de que los campos existen, si no, añadir valores por defecto
        config.setdefault('lm_studio_ip', '192.168.100.49')
        config.setdefault('lm_studio_port', '1234')
        config.setdefault('openai_api_key', '')
        config.setdefault('google_api_key', '')
        return config

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"Saved config to {CONFIG_FILE}")

def configure_dspy_model():
    config = load_config()
    
    if config['model_type'] == 'lm_studio':
        lm_studio_url = f"http://{config['lm_studio_ip']}:{config['lm_studio_port']}/v1/"
        lm = dspy.LM(
            model="openai/google/gemma-3-4b", # Prefijo 'openai/' para indicar el proveedor a litellm
            api_base=lm_studio_url,
            api_key="lm-studio",
            max_tokens=150,
            temperature=0.0,
            timeout=60
        )
    elif config['model_type'] == 'openai_api':
        api_key = config.get('openai_api_key')
        if not api_key:
            raise ValueError("OpenAI API Key no configurada en config.json")
        lm = dspy.LM(
            model='gpt-3.5-turbo', # Puedes cambiar a otro modelo de OpenAI si lo deseas
            api_key=api_key,
            max_tokens=150,
            temperature=0.0,
            timeout=60
        )
    elif config['model_type'] == 'google_api':
        api_key = config.get('google_api_key')
        print(f"Using Google API. API Key: {api_key[:5]}...") # Print first 5 chars of API key for debug
        if not api_key:
            raise ValueError("Google API Key no configurada en config.json")
        lm = dspy.LM(
            model='gemini/gemini-1.5-flash',
            api_key=api_key,
            max_tokens=150,
            temperature=0.0
        )
    else:
        raise ValueError("Tipo de modelo no válido en config.json")
    
    dspy.settings.configure(lm=lm)
    print(f"DSPy LM configured: {lm}")

configure_dspy_model() # Configure DSPy model globally

# --- Definición de la Tarea de Clasificación con una "Signature" de DSPy ---

class ClassifyAd(dspy.Signature):
    """Clasifica un texto como 'anuncio' o 'no_anuncio' y explica brevemente por qué."""

    text_input = dspy.InputField(desc="El texto que necesita ser clasificado.")
    classification = dspy.OutputField(desc="La etiqueta, que debe ser 'anuncio' o 'no_anuncio'.")
    explanation = dspy.OutputField(desc="Una breve explicación de por qué se eligió esa etiqueta.")

# --- Funciones para manejar ejemplos --- 
EXAMPLES_FILE = 'examples.json'
print(f"DEBUG: EXAMPLES_FILE (relative): {EXAMPLES_FILE}")
print(f"DEBUG: Current working directory: {os.getcwd()}")
print(f"DEBUG: EXAMPLES_FILE (absolute): {os.path.abspath(EXAMPLES_FILE)}")

def load_examples():
    examples_data = []
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {EXAMPLES_FILE}: {e}. Initializing with default examples.")
            examples_data = [] # Reset to empty if loading fails

    if not examples_data: # If file didn't exist, was empty, or had an error
        initial_examples_data = [
            {"text_input": "¡50% de descuento en zapatillas, solo hoy!", "classification": "anuncio", "explanation": "Contiene una oferta comercial con un descuento y un sentido de urgencia."},
            {"text_input": "Nos vemos a las 6 en el café.", "classification": "no_anuncio", "explanation": "Es un mensaje personal para coordinar una cita, sin intención comercial."},
            {"text_input": "Gana un viaje gratis, regístrate ahora.", "classification": "anuncio", "explanation": "Promete un premio a cambio de una acción (registrarse), típico de una campaña publicitaria."},
            {"text_input": "Oye, ¿ya terminaste la tarea?", "classification": "no_anuncio", "explanation": "Es una pregunta personal y cotidiana, sin fines comerciales."},
            {"text_input": "Descubre el nuevo smartphone con cámara de 200MP. ¡Resérvalo ya!", "classification": "anuncio", "explanation": "Describe un producto y llama a la acción para comprarlo o reservarlo."},
        ]
        with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
            json.dump(initial_examples_data, f, ensure_ascii=False, indent=4)
        print(f"Created initial {EXAMPLES_FILE}")
        examples_data = initial_examples_data
    
    dspy_examples = []
    for ex in examples_data:
        dspy_examples.append(dspy.Example(
            text_input=ex["text_input"],
            classification=ex["classification"],
            explanation=ex["explanation"]
        ))
    print(f"Loaded {len(dspy_examples)} examples from {EXAMPLES_FILE}")
    return dspy_examples

def save_example(text, classification, explanation):
    examples_data = []
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
            print(f"Read {len(examples_data)} examples from {EXAMPLES_FILE} before appending.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {EXAMPLES_FILE} for saving: {e}. Starting with empty list.")
            examples_data = []
    else:
        print(f"{EXAMPLES_FILE} does not exist before saving. Starting with empty list.")

    examples_data.append({
        "text_input": text,
        "classification": classification,
        "explanation": explanation
    })
    
    print(f"Attempting to save {len(examples_data)} examples to {EXAMPLES_FILE}.")
    with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(examples_data, f, ensure_ascii=False, indent=4)
    print(f"Successfully saved {len(examples_data)} examples to {EXAMPLES_FILE}.")

# --- Rutas de la Aplicación Web ---

@app.route('/', methods=['GET', 'POST'])
def home():
    loaded_examples = load_examples() # Cargar ejemplos en cada petición
    current_config = load_config() # Cargar la configuración actual para mostrarla en la UI
    
    # Crear el programa de clasificación sin pasar los ejemplos (temporalmente)
    classify_program = dspy.Predict(ClassifyAd)

    if request.method == 'POST':
        user_text = request.form['text']
        
        if not user_text.strip():
            return render_template('index.html', error="Por favor, escribe un texto para clasificar.", classification=None, explanation=None, current_config=current_config)

        print(f"Current model configuration: {current_config}")
        try:
            prediction = classify_program(text_input=user_text)
            print(f"Prediction object: {prediction}")
            
            classification_result = prediction.classification if prediction and prediction.classification else "Error: No se pudo clasificar"
            explanation_result = prediction.explanation if prediction and prediction.explanation else "El modelo no proporcionó una explicación."

            print(f"Prediction classification: {classification_result}")
            print(f"Prediction explanation: {explanation_result}")

        except Exception as e:
            print(f"Error during classification: {e}")
            classification_result = "Error"
            explanation_result = f"Error al clasificar: {e}"

        return render_template(
            'index.html',
            classification=classification_result,
            explanation=explanation_result,
            user_text=user_text,
            current_config=current_config,
            loaded_examples=loaded_examples
        )
    
    return render_template('index.html', current_config=current_config, loaded_examples=loaded_examples)

@app.route('/add_example', methods=['POST'])
def add_example():
    example_text = request.form['example_text']
    example_classification = request.form['example_classification']
    example_explanation = request.form['example_explanation']
    
    if not example_text.strip() or not example_explanation.strip():
        return render_template('index.html', error="Por favor, completa todos los campos del ejemplo.", classification=None, explanation=None)
    
    save_example(example_text, example_classification, example_explanation)
    
    return redirect(url_for('home'))

@app.route('/configure_model', methods=['POST'])
def configure_model():
    model_type = request.form['model_type']
    lm_studio_ip = request.form.get('lm_studio_ip', '')
    lm_studio_port = request.form.get('lm_studio_port', '')
    openai_api_key = request.form.get('openai_api_key', '')
    google_api_key = request.form.get('google_api_key', '')

    new_config = {
        'model_type': model_type,
        'lm_studio_ip': lm_studio_ip,
        'lm_studio_port': lm_studio_port,
        'openai_api_key': openai_api_key,
        'google_api_key': google_api_key
    }
    save_config(new_config)
    
    return redirect(url_for('home'))

@app.route('/delete_config', methods=['POST'])
def delete_config():
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"Deleted {CONFIG_FILE}")
    return redirect(url_for('home'))

@app.route('/delete_example', methods=['POST'])
def delete_example():
    text_to_delete = request.form['text_to_delete']
    examples_data = []
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {EXAMPLES_FILE} for deletion: {e}")
            examples_data = []

    updated_examples = [ex for ex in examples_data if ex["text_input"] != text_to_delete]

    with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(updated_examples, f, ensure_ascii=False, indent=4)
    print(f"Deleted example with text: {text_to_delete}. Remaining examples: {len(updated_examples)}")
    return redirect(url_for('home'))

@app.route('/update_example', methods=['POST'])
def update_example():
    original_text = request.form['original_text_input']
    new_text = request.form['text_input']
    new_classification = request.form['classification']
    new_explanation = request.form['explanation']

    examples_data = []
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {EXAMPLES_FILE} for update: {e}")
            examples_data = []

    found = False
    for i, ex in enumerate(examples_data):
        if ex["text_input"] == original_text:
            examples_data[i] = {
                "text_input": new_text,
                "classification": new_classification,
                "explanation": new_explanation
            }
            found = True
            break

    if found:
        with open(EXAMPLES_FILE, 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=4)
        print(f"Updated example from '{original_text}' to '{new_text}'.")
    else:
        print(f"Example with original text '{original_text}' not found for update.")

    return redirect(url_for('home'))

    return redirect(url_for('home'))

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    base64_image = data.get('image')

    if not base64_image:
        return jsonify({'status': 'error', 'message': 'No se recibió ninguna imagen.'})

    config = load_config()
    model_type = config.get('model_type')
    description = ""
    error_message = ""

    try:
        if model_type == 'google_api':
            api_key = config.get('google_api_key')
            if not api_key:
                raise ValueError("Google API Key no configurada.")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Decodificar la imagen para Google Generative AI
            import base64
            from PIL import Image
            import io

            image_bytes = base64.b64decode(base64_image)
            image_pil = Image.open(io.BytesIO(image_bytes))

            response = model.generate_content([
                "Describe esta imagen de forma concisa y objetiva, enfocándote en elementos que puedan indicar si es un anuncio o no. Proporciona *solo* la descripción, sin ninguna introducción, preámbulo o frase inicial.",
                image_pil
            ])
            description = response.text

        elif model_type == 'openai_api':
            api_key = config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API Key no configurada.")
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o", # O gpt-4-vision-preview
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe esta imagen de forma concisa y objetiva, enfocándote en elementos que puedan indicar si es un anuncio o no. Proporciona *solo* la descripción, sin ninguna introducción, preámbulo o frase inicial."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=150
            )
            description = response.choices[0].message.content

        elif model_type == 'lm_studio':
            lm_studio_ip = config.get('lm_studio_ip')
            lm_studio_port = config.get('lm_studio_port')
            if not lm_studio_ip or not lm_studio_port:
                raise ValueError("IP y Puerto de LM Studio no configurados.")
            
            lm_studio_url = f"http://{lm_studio_ip}:{lm_studio_port}/v1/"
            
            client = openai.OpenAI(base_url=lm_studio_url, api_key="lm-studio") # LM Studio usa la API de OpenAI
            
            response = client.chat.completions.create(
                model="google/gemma-3-4b", # Usar el nombre de modelo correcto para LM Studio
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe esta imagen de forma concisa y objetiva, enfocándote en elementos que puedan indicar si es un anuncio o no. Proporciona *solo* la descripción, sin ninguna introducción, preámbulo o frase inicial."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=150
            )
            description = response.choices[0].message.content
        else:
            error_message = "Tipo de modelo no válido para procesamiento de imágenes."

    except Exception as e:
        error_message = f"Error al procesar la imagen con el modelo {model_type}: {str(e)}"
        print(f"Error processing image: {e}")

    if description:
        return jsonify({'status': 'success', 'description': description})
    else:
        return jsonify({'status': 'error', 'message': error_message or "No se pudo generar una descripción de la imagen."})

@app.route('/test_lm_studio_connection', methods=['POST'])
def test_lm_studio_connection():
    ip = request.form.get('lm_studio_ip')
    port = request.form.get('lm_studio_port')
    
    if not ip or not port:
        return jsonify({'status': 'error', 'message': 'IP y Puerto son requeridos.'})

    try:
        test_url = f"http://{ip}:{port}/v1/models"
        response = requests.get(test_url, timeout=5) # Timeout de 5 segundos
        
        if response.status_code == 200:
            # Intentar parsear JSON para ver si es una respuesta válida de la API de OpenAI
            try:
                models_data = response.json()
                if "data" in models_data and isinstance(models_data["data"], list):
                    return jsonify({'status': 'success', 'message': 'Conexión exitosa a LM Studio. Modelos disponibles: ' + ', '.join([m['id'] for m in models_data['data']])})
                else:
                    return jsonify({'status': 'warning', 'message': 'Conexión exitosa, pero la respuesta de la API no es la esperada.'})
            except json.JSONDecodeError:
                return jsonify({'status': 'warning', 'message': 'Conexión exitosa, pero la respuesta no es JSON válido.'})
        else:
            return jsonify({'status': 'error', 'message': f'Error de conexión: Código de estado {response.status_code}.'})
    except requests.exceptions.ConnectionError:
        return jsonify({'status': 'error', 'message': 'No se pudo conectar a LM Studio. Verifica la IP y el Puerto.'})
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'message': 'Tiempo de espera agotado al conectar con LM Studio.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error inesperado: {str(e)}'})

# --- Punto de Entrada para Correr la App ---
if __name__ == '__main__':
    app.run(debug=True)
