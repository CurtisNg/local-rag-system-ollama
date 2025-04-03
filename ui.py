import streamlit as st
from processor import DocumentProcessor, get_available_models
from PIL import Image
import torch

# Check if MPS is available on M1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Set page configuration
st.set_page_config(page_title="Ollama AI Chatbot", layout="wide")
st.title("💬 Ollama AI Chatbot")

# Sidebar setup
st.sidebar.header("Select Mode")
page = st.sidebar.selectbox("Pick a chat:", ["Document Chat", "Image-based Chat"])

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["doc_chat"] = []
    st.session_state["img_chat"] = []
    st.success("Chat history cleared!")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a document for Q&A", type=["pdf", "csv"]) if page == "Document Chat" else None
uploaded_image = st.sidebar.file_uploader("Upload an image for analysis (optional)", type=["png", "jpg", "jpeg"]) if page == "Image-based Chat" else None

# Model selection
st.sidebar.header("Select Model")
available_models = get_available_models()
vision_models = ['llava', 'gemma']

if page == "Image-based Chat":
    img_model = [m for m in available_models if any(vm in m.lower() for vm in vision_models)]
    if not img_model:
        st.sidebar.warning("No LLaVA model found! Download a model to proceed.")
        st.stop()
    selected_model = st.sidebar.selectbox("Pick a model:", img_model)
else:
    if not available_models:
        st.sidebar.warning("No models found! Download a model to proceed.")
        st.stop()
    selected_model = st.sidebar.selectbox("Pick a model:", available_models)

# Initialize processor
processor = DocumentProcessor(selected_model)

# Initialize chat histories
if "doc_chat" not in st.session_state:
    st.session_state["doc_chat"] = []
if "img_chat" not in st.session_state:
    st.session_state["img_chat"] = []

# Process uploaded files
retriever, full_text, csv_df, csv_summary, csv_data = None, "", None, "", ""
if page == "Document Chat" and uploaded_file:
    if uploaded_file.type == "application/pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        retriever, full_text = processor.process_pdf("temp.pdf")
        st.success("PDF processed! Start asking questions.")
    elif uploaded_file.type == "text/csv":
        csv_df, csv_summary, csv_data = processor.process_csv(uploaded_file)
        st.success("CSV file processed! Start asking questions.")

# Document Chat UI
if page == "Document Chat":
    st.subheader(f"📁 Document Chat - {selected_model}")
    
    # if csv_data:
    #     st.write("**Uploaded CSV File Preview**")
    #     st.write(f"Total Rows: {csv_df.shape[0]} | Total Columns: {csv_df.shape[1]}")
    #     st.dataframe(csv_df.head())

    for message in st.session_state["doc_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state["doc_chat"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Thinking..."):
            response = processor.answer_question(
                user_input, retriever=retriever, csv_summary=csv_summary, csv_data=csv_data, full_text=full_text
            )
            st.session_state["doc_chat"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# Image-based Chat UI
if page == "Image-based Chat":
    st.subheader(f"🖼️ Image Chat - {selected_model}")
    
    for message in st.session_state["img_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        max_width = 250
        if image.size[0] > max_width:
            aspect_ratio = image.size[1] / image.size[0]
            image = image.resize((max_width, int(max_width * aspect_ratio)))
        st.image(image, caption="Uploaded Image", use_container_width=False)

    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state["img_chat"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Thinking..."):
            if uploaded_image:
                response = processor.process_image(Image.open(uploaded_image), user_input)
            else:
                response = processor.llm.invoke(user_input)
            st.session_state["img_chat"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)