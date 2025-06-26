# Function to process user input and return appropriate response
def chatbot_response(message):
    message = message.lower()  # Convert input to lowercase for easy matching

    if message in ["hi", "hello", "hey"]:
        return "Hello! I'm your virtual assistant. How can I help you today?"
    elif "your name" in message:
        return "I'm ChatBotX, your simple rule-based assistant."
    elif "how are you" in message:
        return "I'm just code, but I'm functioning perfectly. Thanks for asking!"
    elif "help" in message:
        return "Sure, I can help! Try asking me about my name, or say hello."
    elif "bye" in message or "exit" in message:
        return "Goodbye! Wishing you a great day ahead!"
    else:
        return "Sorry, I didn't understand that. Can you try saying it differently?"

# Main chat loop to interact with the user
if __name__ == "__main__":
    print("ChatBotX: Hello! Type 'bye' to end the chat.")
    while True:
        user_input = input("You: ")
        response = chatbot_response(user_input)
        print("ChatBotX:", response)
        if "bye" in user_input.lower() or "exit" in user_input.lower():
            break
