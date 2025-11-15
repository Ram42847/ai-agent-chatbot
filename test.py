import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyBbJ5uZplbwRZs53LRjXlv5pRiy9-vKk6M")  # <-- Set the API key here


#models = genai.list_models()
#for model in models:
#    print(model.name)


model = genai.GenerativeModel("models/gemini-2.5-flash")   # or whatever model you have access to

response = model.generate_content("Explain how AI works in a few words")

print(response.text)

