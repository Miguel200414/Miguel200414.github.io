from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from validators import url as validate_url
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'  # Ejemplo con SQLite, puedes cambiar la URI según tu base de datos
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutos de tiempo de caché
cache = Cache(app)
db = SQLAlchemy(app)

tokenizer = AutoTokenizer.from_pretrained("gpt-3.5-turbo")
model = AutoModelForCausalLM.from_pretrained("gpt-3.5-turbo")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_INPUT_LENGTH = 1000
MAX_RESPONSE_LENGTH = 500
MAX_CHAT_HISTORY_LENGTH = 3  # Número máximo de interacciones de chat en el historial

chat_history = []

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)

# Inicialización
model = model.to(device)
model.eval()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if msg:
        response = get_chat_response(msg)
        return response
    return ""

@cache.memoize()
def get_chat_response(text):
    global chat_history
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)

    # Comprobar si el mensaje es un comentario
    if text.lower().startswith("feedback:"):
        process_feedback(text)
        return "Gracias por tu comentario."

    # Validar la longitud mínima y máxima de entrada
    if len(inputs[0]) < 2 or len(inputs[0]) > MAX_INPUT_LENGTH:
        return "Por favor, introduce una consulta válida."

    # Limitar el historial de chat a un máximo de MAX_CHAT_HISTORY_LENGTH interacciones
    if len(chat_history) >= MAX_CHAT_HISTORY_LENGTH:
        chat_history = chat_history[1:]

    # Concatenar la entrada del usuario con el historial de chat
    bot_input_ids = torch.cat([chat["input_ids"] for chat in chat_history] + [inputs], dim=-1)

    # Resto del código para generar la respuesta...
    response = generate_response(bot_input_ids)

    # Actualizar el historial de chat con la entrada del usuario y la respuesta generada
    chat_history.append({"input_ids": inputs, "response": response})

    return response

def post_process_response(response):
    # Filtrar respuestas inapropiadas
    if "inappropriate" in response:
        response = "Lo siento, no puedo proporcionar una respuesta adecuada en este momento."

    # Mejorar la legibilidad de la respuesta
    translation_map = str.maketrans({"'": None})
    response = response.translate(translation_map)
    response = response.replace("n't", " no").replace("'m", " soy").replace("'re", " eres").replace("'ll", " lo haré")

    # Capitalizar la primera letra de la respuesta
    response = response.capitalize()

    return response

def process_feedback(feedback):
    # Obtener el contenido del comentario eliminando el prefijo
    feedback_content = feedback[len("feedback:"):].strip()

    # Guardar el comentario en la base de datos
    save_feedback(feedback_content)

    # Devolver un mensaje de confirmación
    return "Gracias por tu comentario. Hemos registrado tu comentario."

def save_feedback(feedback_content):
    feedback = Feedback(content=feedback_content)
    db.session.add(feedback)
    db.session.commit()

def generate_response(bot_input_ids):
    with torch.no_grad():
        model_outputs = model.generate(
            bot_input_ids,
            max_length=MAX_INPUT_LENGTH + MAX_RESPONSE_LENGTH,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,  # Aumentar la temperatura para obtener más diversidad
            num_return_sequences=1
        )
        response = tokenizer.decode(model_outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        response = response[-MAX_RESPONSE_LENGTH:]  # Limitar la longitud de la respuesta generada
        response = post_process_response(response)
        return response

def get_text_from_url(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

@app.route("/feed", methods=["POST"])
def feed():
    url = request.form.get("url")
    if url:
        if not validate_url(url):
            return "La URL proporcionada no es válida. Por favor, verifica la URL y asegúrate de que es accesible."

        text = get_text_from_url(url)
        if text:
            response = get_chat_response(text)
            return response
        else:
            return "No se pudo obtener el contenido de la URL. Por favor, verifica la URL y asegúrate de que es accesible."
    return ""

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
