import google.generativeai as genai

# Configure API key
genai.configure(api_key="xxx-xxx-xx")  

model = genai.GenerativeModel("gemini-1.5-flash")
def chat_with_gemini():
    print("Chatbot (Gemini 1.5 Flash). Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        response = model.generate_content(user_input)
        print("Chatbot:", response.text)

if __name__ == "__main__":
    chat_with_gemini()

