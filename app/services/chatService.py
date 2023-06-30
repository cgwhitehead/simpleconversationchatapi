from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def prepare_bot_input_ids(chat_history_ids, new_user_input_ids):
    """
    Prepares bot_input_ids by concatenating the chat history with the new user input.

    Args:
        chat_history_ids (torch.Tensor): Tensor representing the chat history.
        new_user_input_ids (torch.Tensor): Tensor representing the new user input.

    Returns:
        torch.Tensor: Concatenated tensor of chat history and new user input.
    """
    if chat_history_ids is None:
        # If there is no chat history, return the new user input as is
        return new_user_input_ids

    # Concatenate the chat history with the new user input
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    return bot_input_ids


def save_question_to_text_file(input_sentence, filename):
    """
    Saves the input sentence to a text file.

    Args:
        input_sentence (str): Input sentence to be saved.
        filename (str): Name of the text file to save the sentence.
    """
    with open(filename, 'a') as file:
        file.write(input_sentence)


def read_from_text_file(filename):
    """
    Reads the contents of a text file.

    Args:
        filename (str): Name of the text file to read.

    Returns:
        str: Contents of the text file.
    """
    chat_history_text = ""
    with open(filename, 'r') as file:
        for line in file:
            chat_history_text += line.strip()
    return chat_history_text


class ChatService:
    def __init__(self, app_config):
        model_path = app_config['PATH_TO_MODEL']
        "-GPT2-Large/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.chat_history_ids = None
        self.file_as_db_location = app_config['PATH_TO_FILE_DB']


    def chat_bot_service(self, input_sentence):
        """
        Provides the chat bot service by generating a response to the user input.

        Args:
            input_sentence (str): User input sentence.

        Returns:
            str: Generated response from the chat bot.
        """
        old_chats = read_from_text_file(self.file_as_db_location)
        tokenized_old_chats = self.tokenize_user_input(old_chats)
        tokenized_user_input = self.tokenize_user_input(input_sentence)
        save_question_to_text_file(input_sentence, self.file_as_db_location)
        prepare_bot_input_ids(self.chat_history_ids, tokenized_old_chats)
        bot_input_ids = prepare_bot_input_ids(self.chat_history_ids, tokenized_user_input)
        self.chat_history_ids = self.add_to_history(bot_input_ids)
        self.save_answer_to_text_file(bot_input_ids, self.file_as_db_location)
        return self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    def tokenize_user_input(self, sentence_string):
        """
        Tokenizes the user input sentence.

        Args:
            sentence_string (str): User input sentence.

        Returns:
            torch.Tensor: Tokenized user input.
        """
        new_user_input_ids = self.tokenizer.encode(sentence_string + self.tokenizer.eos_token, return_tensors='pt')
        return new_user_input_ids

    def add_to_history(self, bot_input_ids):
        """
        Adds the bot input to the chat history.

        Args:
            bot_input_ids (torch.Tensor): Bot input tensor.

        Returns:
            torch.Tensor: Updated chat history tensor.
        """
        chat_history_ids = self.model.generate(bot_input_ids, max_length=2000, pad_token_id=self.tokenizer.eos_token_id)
        return chat_history_ids

    def save_answer_to_text_file(self, bot_input_ids, filename):
        """
        Saves the generated answer to a text file.

        Args:
            bot_input_ids (torch.Tensor): Bot input tensor.
            filename (str): Name of the text file to save the answer.
        """
        with open(filename, 'a') as file:
            file.write(
                self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
