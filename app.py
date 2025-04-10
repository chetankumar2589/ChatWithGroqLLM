import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import gradio as gr

# Load API key from environment (use Hugging Face secret)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

template = """Meet Bantu, your youthful and witty personal assistant! At 21 years old, he's full of energy and always eager to help. Bantu's goal is to assist you with any questions or problems you might have. His enthusiasm shines through in every response, making interactions with him enjoyable and engaging.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"],
    template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.5,
        model_name="llama-3.3-70b-versatile"
    ),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

def get_text_response(user_message, history):
    return llm_chain.run(user_message=user_message)

demo = gr.ChatInterface(
    fn=get_text_response,
    examples=[
        "How are you doing?",
        "What are your interests?",
        "Which places do you like to visit?"
    ],
    title="💬 Chat with Bantu – Powered by Groq + LLaMA 3.3",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
