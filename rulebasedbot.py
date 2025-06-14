import re
import nltk
from nltk.chat.util import Chat,reflections
nltk.download('averaged_perceptron_tagger')

class RuleBasedChatBot:
    def __init__(self,pairs):
        self.chat = Chat(pairs,reflections)

    def respond(self,user_input):
        return self.chat.respond(user_input)
    
def chat_with_bot():
    print("Welcome to the virtual world !! type 'exit' to exit")
    while True:
        ui = input("You : ")
        if ui.lower() == 'exit':
            print("Chatbot : Goodbye Master !! Hope you had agreat time talking with me !!")
            break
        response = chatbot.respond(ui)
        print(f"Chatbot : {response}")
    

#For pattern matching we need to define pairs which include user input and response
pairs = [
    [r"[hH](i|ello|ey)",["Hello! How can I help you today?", "Hi there! How may I help you ?"]],
    [r"my name is (.*)", ["Hello %1! How can I assist you today?"]],
    [r"(.*) your name?", ["I am your friendly chatbot!"]],
    [r"how are you?", ["I'm just a bot, but I'm doing well. How about you?"]],
    [r"tell me a joke", ["Why don't skeletons fight each other? They don't have the guts!"]],
    [r"(.*) (help|assist) (.*)", ["Sure! How can I assist you with %3?"]],
    [r"bye|exit|See You |Goodnight|ttyl", ["Goodbye! Have a great day!", "See you later!"]],
    [r"(.*)", ["I'm sorry, I didn't understand that. Could you rephrase?", "Could you please elaborate?"]]
]
chatbot = RuleBasedChatBot(pairs)
chat_with_bot()
