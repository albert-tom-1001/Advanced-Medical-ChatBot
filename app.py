import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import spacy

# Function to get environment variables
def get_env_variable(var_name):
    if 'STREAMLIT_RUNTIME_ENV' in os.environ:
        # We're running on Streamlit Cloud
        return st.secrets[var_name]
    else:
        # We're running locally
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(var_name)

# Set API keys
openai_api_key = get_env_variable("OPENAI_API_KEY")
groq_api_key = get_env_variable("GROQ_API_KEY")

# Set environment variables
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a folder for storing embeddings if it doesn't exist
EMBED_STORE_DIR = "embed_store"
if not os.path.exists(EMBED_STORE_DIR):
    os.makedirs(EMBED_STORE_DIR)

# Set page config
st.set_page_config(
    page_title="MediChat AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Force light theme
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f0f4f8 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS
custom_css = """
<style>
.stApp {
    background-color: #f0f4f8;
    color: #262730;
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.st-bw {
    background-color: #ffffff;
}
.st-eb {
    background-color: #5c9bbf;
    color: white;
}
.st-eb:hover {
    background-color: #4a7d9d;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #e6f3ff;
    color: #262730;
}
.chat-message.bot {
    background-color: #f0f0f0;
    color: #262730;
}
.chat-icon {
    width: 50px;
    height: 50px;
    margin-right: 1rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}
.user .chat-icon {
    background-color: #4a7d9d;
    color: white;
}
.bot .chat-icon {
    background-color: #5c9bbf;
    color: white;
}
.chat-content {
    flex-grow: 1;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    color: #262730 !important;
    background-color: #ffffff !important;
    border: 1px solid #cccccc !important;
    caret-color: #262730 !important;
}
body, input, textarea, button {
    color: #262730 !important;
}
::placeholder {
    color: #999999 !important;
    opacity: 1 !important;
}
.stMarkdown {
    color: #262730;
}
h1, h2, h3, h4, h5, h6 {
    color: #262730;
}
p {
    color: #262730;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', (event) => {
    const style = document.createElement('style');
    style.textContent = `
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            color: #262730 !important;
            background-color: #ffffff !important;
            border: 1px solid #cccccc !important;
            caret-color: #262730 !important;
        }
    `;
    document.head.appendChild(style);

    const updateInputStyles = () => {
        const inputs = document.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.style.color = '#262730';
            input.style.backgroundColor = '#ffffff';
        });
    };

    updateInputStyles();
    setInterval(updateInputStyles, 1000);
});
@keyframes blink {
    0% { opacity: 0; }
    50% { opacity: 1; }
    100% { opacity: 0; }
}
.blinking-cursor {
    animation: blink 1s step-end infinite;
    display: inline-block;
}
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    session_vars = [
        "vectors", "embeddings", "chain", "conversation_history",
        "current_response"
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {
            "topic": None,
            "entities": [],
            "user_preferences": {},
            "previous_queries": []
        }
    
    if "typing_speed" not in st.session_state:
        st.session_state.typing_speed = 0.05  # Default typing speed

# Initialize session state
initialize_session_state()

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

llm = get_llm()

# Define the enhanced chat prompt template
prompt = ChatPromptTemplate.from_template(
"""
You are an advanced AI medical assistant designed to provide comprehensive medical information. Your primary function is to offer accurate, helpful, and context-aware responses while maintaining ethical standards and user safety.

Previous Chat History:
{chat_history}

Context from documents:
{context}

Current question: {question}

OPERATIONAL PROTOCOL:

1. DIRECT ANSWERING:
   - Start your response immediately with relevant information.

2. STRUCTURED AND CLEAR RESPONSES:
   - Always use appropriate formatting to enhance readability.
   - Utilize bullet points or numbered lists for steps, symptoms, or multiple pieces of advice.
   - Separate distinct topics or ideas into different paragraphs.
   - For complex topics, use headings to organize information (e.g., "Symptoms:", "Treatment:", "When to Seek Medical Help:").

3. INFORMATION RETRIEVAL AND SYNTHESIS:
   - Analyze the provided context, chat history, and current question.
   - Synthesize information from multiple sources if necessary.

4. CONTEXT-AWARE ANSWERING:
   - Ensure your response is relevant to the current question and previous context if applicable.

5. COMPREHENSIVE MEDICAL GUIDANCE:
   - For questions about symptoms, clearly list possible conditions and when to seek medical attention.
   - When discussing treatments or medications, structure the information logically (e.g., home remedies, over-the-counter options, prescription treatments).


Remember to use appropriate formatting throughout your response to enhance readability and understanding. Do not present all information in a single paragraph.

Proceed with your response following this protocol:
"""
)

# Filter out non-essential sections based on keywords
def filter_documents(documents):
    excluded_sections = ["author", "acknowledgments", "preface", "introduction", "foreword", "references"]
    return [doc for doc in documents if not any(keyword in doc.page_content.lower() for keyword in excluded_sections)]

# Automatically load or create vector store
@st.cache_resource
def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings()
        
        if os.path.exists(f"{EMBED_STORE_DIR}/faiss_index"):
            vectors = FAISS.load_local(f"{EMBED_STORE_DIR}/faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader("./Data/Medical_book.pdf")
            docs = loader.load()
            docs = filter_documents(docs)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, embeddings)
            vectors.save_local(f"{EMBED_STORE_DIR}/faiss_index")
        
        return vectors, embeddings
    except Exception as e:
        st.error(f"An error occurred during vector store loading: {str(e)}")
        return None, None

# Load vector store
vectors, embeddings = load_vector_store()

# Create the conversational retrieval chain
@st.cache_resource
def get_conversation_chain(_llm, _vectors):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vectors.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )

chain = get_conversation_chain(llm, vectors)

# Enhanced topic detection using SpaCy
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

def extract_medical_entities(question):
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "SYMPTOM", "BODY_PART", "MEDICAL_CONDITION", "MEDICATION"]]
    return entities if entities else None

# Improved query preprocessing
def preprocess_query(query):
    general_queries = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what are you", "what can you do", "who made you",
        "what is your purpose", "how does this work"
    ]
    
    query_lower = query.lower()
    
    if any(gen_q in query_lower for gen_q in general_queries):
        return "general", query
    
    medical_entities = extract_medical_entities(query)
    if medical_entities:
        return "medical", query
    
    return "unknown", query

# Handle general queries
def handle_general_query(query):
    responses = {
        "hello": "Hello! I'm MediChat AI, your advanced medical assistant. How can I help you with health-related questions today?",
        "how are you": "As an AI, I'm always ready to assist. How can I help with your medical questions or concerns?",
        "what can you do": "I can provide information on medical conditions, symptoms, treatments, and general health advice. What specific topic would you like to explore?",
        "who made you": "I was developed by a team of AI specialists and medical professionals. How may I assist you with health-related inquiries?",
        "what is your purpose": "My purpose is to provide accurate medical information and guide users towards appropriate healthcare resources. What health topic shall we discuss?",
        "how does this work": "You can ask me about any medical condition, symptom, treatment, or health-related topic. I'll provide information based on my training. What health concern would you like to explore?"
    }
    
    for key, response in responses.items():
        if key in query.lower():
            return response
    
    return "I'm here to assist with medical questions and health concerns. What specific medical topic would you like to discuss?"

# Improved query processing and context management
def process_query(query):
    query_type, processed_query = preprocess_query(query)

    if query_type == "general":
        return handle_general_query(processed_query), []
    else:
        medical_entities = extract_medical_entities(processed_query)
        
        # Update conversation context
        st.session_state.conversation_context["previous_queries"].append(processed_query)
        if medical_entities:
            st.session_state.conversation_context["entities"] = list(set(st.session_state.conversation_context["entities"] + medical_entities))
            st.session_state.conversation_context["topic"] = medical_entities[0]  # Assume the first entity is the main topic
        
        # If no entities are found in the current query, use the context from previous queries
        if not medical_entities and st.session_state.conversation_context["entities"]:
            medical_entities = st.session_state.conversation_context["entities"]
        
        context_info = (
            f"Current topic: {st.session_state.conversation_context.get('topic', 'Not set')}. "
            f"Related entities: {', '.join(st.session_state.conversation_context.get('entities', []))}. "
            f"User preferences: {st.session_state.conversation_context.get('user_preferences', {})}. "
            f"Previous queries: {', '.join(st.session_state.conversation_context['previous_queries'][-3:])}"
        )
        
        # Combine context_info and processed_query into a single input
        combined_query = f"{context_info} Question: {processed_query}"

        response = chain({"question": combined_query})
        
        # Preserve formatting in the response
        formatted_response = response['answer'].replace('\n', '<br>')
        
        # Add a brief disclaimer
        disclaimer = "<br><br>Disclaimer: Consult a healthcare professional for personalized medical advice."
        
        return f"<div style='color: #262730;'>{formatted_response}</div>" + disclaimer, response.get('source_documents', [])
    
# User input validation
def validate_user_input(user_input):
    if not user_input.strip():
        return False, "Please enter a valid question or command."
    if len(user_input) > 500:
        return False, "Your input is too long. Please limit your question to 500 characters."
    return True, ""

# Streamlit UI
st.markdown("<h1 style='color: #262730;'>🏥 MediChat AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #262730;'>Your Advanced Medical Assistant</h3>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div style='color: #262730;'>{message['content']}</div>", unsafe_allow_html=True)

# ... (previous code remains the same)

# Typing speed control
typing_speed = st.sidebar.slider(
    "Adjust typing speed",
    min_value=0.01,
    max_value=0.1,
    value=st.session_state.typing_speed,
    step=0.01,
    key="typing_speed_slider"
)
st.session_state.typing_speed = typing_speed


# React to user input
if prompt := st.chat_input("Ask your medical question here"):
    # Validate user input
    is_valid, error_message = validate_user_input(prompt)
    if not is_valid:
        st.error(error_message)
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(f"<div style='color: #262730;'>{prompt}</div>", unsafe_allow_html=True)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with adjustable delay
            with st.spinner("MediChat AI is thinking..."):
                response, source_documents = process_query(prompt)
                words = response.split()
                for i, word in enumerate(words):
                    full_response += word + " "
                    if i % 5 == 0 or i == len(words) - 1:  # Update every 5 words or at the end
                        time.sleep(st.session_state.typing_speed * 5)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(f"{full_response}<span class='blinking-cursor'>|</span>", unsafe_allow_html=True)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        

# Sidebar with app information
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/flat-hand-drawn-hospital-reception-scene_52683-54613.jpg?w=1380&t=st=1691881051~exp=1691881651~hmac=e8ef2148b778d3937b7410ff3a72439b2fb46496da5d37463fb4d374bc7baa1c", use_column_width=True)
    st.title("About MediChat AI")
    st.info(
        "MediChat AI is an advanced medical assistant powered by artificial intelligence. "
        "It provides information on various medical topics, symptoms, and general health advice. "
        "Always consult with a healthcare professional for personalized medical advice and treatment."
    )
    st.warning(
        "⚠️ Disclaimer: MediChat AI is for informational purposes only and should not be considered as a substitute for professional medical advice, diagnosis, or treatment."
    )