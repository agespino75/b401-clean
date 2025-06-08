import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI

llmModel = OpenAI()


from langchain_openai import ChatOpenAI

chatModel = ChatOpenAI(model="gpt-4o-mini")


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
    {"input": "how are you?", "output": "¿cómo estás?"},
    {"input": "what's your name?", "output": "¿cuál es tu nombre?"},
    {"input": "I love programming.", "output": "me encanta programar."},
    {"input": "What is the capital of France?", "output": "¿cuál es la capital de Francia?"},
    {"input": "Translate 'good morning' to Spanish.", "output": "traduce 'buenos días' al español."},
    {"input": "Translate 'good night' to Spanish.", "output": "traduce 'buenas noches' al español."},

]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | chatModel

response = chain.invoke({"input": "Who was JFK?"})

print("\n----------\n")

print("Translate: Who was JFK?")
print(response.content)
print("\n----------\n")
print("Translate: Who was JFK?")
print("Response:", response.content if hasattr(response, "content") else response)
print("\n----------\n")