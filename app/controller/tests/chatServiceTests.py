import unittest
from unittest.mock import patch, MagicMock

from flask import Flask

from app.config import load_config
from app.services.chatService import ChatService


class TestChatService(unittest.TestCase):
    def setUp(self):
        #mock_config = MagicMock()
        #mock_config.DEBUG = True
        #mock_config.SECRET_KEY = 'mysecretkey'
        ## Set other configuration values as needed
#
        ## Assign the mock config to the app.config
        #app.config = mock_config
        app = Flask(__name__)
        load_config(app)
        self.chat_service = ChatService(app.config)

    def test_chat_bot_service(self):
        input_sentence = "Hello"
        expected_response = "Expected response from the chat bot"

        with patch.object(self.chat_service, 'chat_bot_service', return_value=expected_response):
            response = self.chat_service.chat_bot_service(input_sentence)

        self.assertEqual(response, expected_response)

    def test_tokenize_user_input(self):
        sentence_string = "Hello"
        expected_tokenized_input = "Expected tokenized input"

        with patch.object(self.chat_service.tokenizer, 'encode', return_value=expected_tokenized_input):
            tokenized_input = self.chat_service.tokenize_user_input(sentence_string)

        self.assertEqual(tokenized_input, expected_tokenized_input)

    def test_add_to_history(self):
        bot_input_ids = "Bot input IDs"
        expected_chat_history_ids = "Expected chat history IDs"

        with patch.object(self.chat_service.model, 'generate', return_value=expected_chat_history_ids):
            chat_history_ids = self.chat_service.add_to_history(bot_input_ids)

        self.assertEqual(chat_history_ids, expected_chat_history_ids)


if __name__ == '__main__':
    unittest.main()