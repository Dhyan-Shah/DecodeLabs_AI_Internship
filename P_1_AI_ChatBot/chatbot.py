# =======================================================
# Decode_Lab | AI_Internship (Virtual_1_Month)
# Dhyan Shah 
# Project_1
# Rule_Based_AI_Chatbot
# =======================================================

RESPONSES={
    #Greetings
    "hello":"Hey there! I'm DecoBot . How can I help you today?",
    "hi":"Hi! Great to see you, What's on your mind?",
    "hey":"Hey! DecoBot at your service. How can I assist you today?",

    #Identity
    "Who are you":"I'm DecoBot- a rule based AI chatbot",
    "What are you? ":"I'm a rule-based AI chatbot designed to assist you with information and tasks. I use dictionary of intents to reliable responses",
    "your name":"My name is DecoBot. I'm here to help you with any questions or tasks you have.",

    #AI concepts
    "what is AI?":"AI stands for Artificial Intelligence. It refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
    "what is ml":"Machine Learning is a subset of AI that focuses on the development of algorithms that can learn from and make predictions or decisions based on data.",
    "what is deep learning":"Deep Learning is a subset of Machine Learning that uses neural networks with many layers (hence 'deep') to model and understand complex patterns in data.",    
    "rule based":"A rule-based AI chatbot operates on a set of predefined rules and logic to generate responses.",

    #Help
    "help":"Sure you can ask me about AI concepts, or just have a friendly chat or just say hello. Type 'quit' or 'exit' to end chat! I'm here to assist you with any questions you may have.",

    #Small talk
    "how are you":"I'm just a bunch of code, but I'm here to help you! How can I assist you today?",
    "good morning":"Good morning! Hope you have a fantastic day ahead!",
    "good night":"Good night! Sleep well and have sweet dreams!",


    #Farewell
    "bye":"Goodbye! It was nice chatting with you. If you have any more questions in the future, feel free to ask. Take care!",
    "goodbye":"Goodbye! It was nice chatting with you. If you have any more questions in the future, feel free to ask. Take care!",
    "thank you":"You're welcome! If you have any more questions or need assistance in the future, don't hesitate to ask. Have a great day!",
    "thanks":"You're welcome! If you have any more questions or need assistance in the future, don't hesitate to ask. Have a great day!",
    
}

EXIT_KEYWORDS={"quit", "exit","bye","goodbye","stop","end"}

# -------------PHASE_1:  INPUT SANITIZATION-----------------
def sanitize_input(raw: str)->str:
    # Convert to lowercase
    user_input=raw.lower().strip()
    return user_input

# --------------PHASE_2: INTENT MATCHING-----------------
def get_response(clean_input:str)->str:
    """
    1. Direct O(1) lookup in RESPONSES dictionary
    2. If no found, keyword scan(partial matching)
    3. Fallback for completely unknown input
    """

    # Direct match
    if clean_input in RESPONSES:
        return RESPONSES[clean_input]
    
    # Keyword scan- checks if any known keyword is present in user input
    for key in RESPONSES:
        if key in clean_input:
            return RESPONSES[key]
        

    # Fallback response
    return ("I don't understand that yet. My knowledge is rule-based, "
            "so I only know what I've been taught. Try asking about AI, "
            "ML,DL. Type 'help' for options.")


#------------PHASE_3:THE HEARTBEAT LOOP----------------
def run_chatbot():
    print('='*55)
    print("Welcome to DecoBot! Type 'help' for options or 'quit' to exit.")
    print('='*55)

    print()

while True:

    #--------INPUT------------
    user_input=input("You: ")


    #---------SANITIZATION---------
    clean_input=sanitize_input(user_input)

    if clean_input in EXIT_KEYWORDS:
        print("DecoBot: Goodbye! It was nice chatting with you. Take care!")
        break

    #---------EMPTY_INPUT_CHECK---------
    if not clean_input:
        print("DecoBot: It seems you didn't type anything. Please enter a message or type 'help' for options.")
        continue


    #---------RESPONSE_GENERATION---------
    response=get_response(clean_input)
    print(f"DecoBot: {response}")


#--------ENTRY POINT---------
if __name__=="__main__":
    run_chatbot()
