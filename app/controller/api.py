import jsonschema
from flask import Flask, request, jsonify
from flask_expects_json import expects_json
from app.services.chatService import ChatService
from app.config import load_config

app = Flask(__name__)
load_config(app)


# Define the schema for the JSON payload
schema = {
    'type': 'object',
    'properties': {
        'sentence': {'type': 'string'}
    },
    'required': ['sentence']
}


@app.route('/chatbot/conversational', methods=['POST'])
@expects_json(schema)
def endpoint():
    """
    API endpoint for the conversational chat service.
    Returns:
        tuple: JSON response and HTTP status code.
    """
    # Get the request data
    data = request.get_json()
    input_sentence = data.get('sentence')
    chat_service = ChatService(app.config)
    chat_response = "Bot: {}".format(chat_service.chat_bot_service(input_sentence))
    # Return a response with chat_response included
    response = {'message': 'Success', 'chat_response': chat_response}
    return jsonify(response), 200


@app.errorhandler(jsonschema.ValidationError)
def on_validation_error(e):
    """
    Error handler for JSON schema validation errors.
    Args:
        e (jsonschema.ValidationError): Validation error object.
    Returns:
        tuple: JSON response and HTTP status code.
    """
    response = {'error': 'Invalid request payload'}
    return jsonify(response), 400
