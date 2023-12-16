from Answer_Processor import *
from openai import AzureOpenAI
import pandas as pd

client = AzureOpenAI(
    api_key = "7282c43ed0b549118f59d015ccdc467d",  
    api_version = "2023-05-15",
    azure_endpoint = "https://openapiazconf.openai.azure.com/"
    )
deployment_name='prohackathon-2'

def Start_Chat_Bot():

    while True:
        user_input = input("You: ")

        # Check for termination command
        if user_input.lower() in ["stop", "exit"]:
            print("Chat ended.")
            break

        try:
            response = client.chat.completions.create(
            model="prohackathon-2",
            messages=[
                {"role": "system", "content": """Assistant is an intelligent chatbot designed to help users classify the question given by the user.
                Instructions:
                - Answer must be only of three options "descriptive","predictive" or "prescriptive" based on the question.
                - If you're unsure of an answer, you can say as "descriptive" """},
                {"role": "user", "content": user_input}
            ])
            #print(response.model_dump_json(indent=2))
            question_type = response.choices[0].message.content
            print(question_type)
            result = Get_Answer(user_input, question_type)
            print(result)
            #print("Bot:", response.choices[0].text)
        except Exception as e:
            print("An error occurred:", e)
            break
