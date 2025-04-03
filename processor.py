import ollama
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import torch
import base64
from io import BytesIO
from PIL import Image
import re  # For robust text cleanup

# Check if MPS is available on M1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Single flexible prompt
UNIVERSAL_PROMPT = PromptTemplate.from_template("""
You are an expert assistant. Use the following context to answer the question or provide a summary if requested.
Context: {context}
Question: {question}
If summarizing, provide a concise overview of the content.
If you cannot answer or summarize based on the context, say "I don't know based on the provided data."
Helpful Answer:
""")

class DocumentProcessor:
    def __init__(self, model_name):
        self.llm = Ollama(model=model_name)
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", model_kwargs={'device': device})
        self.text_splitter = SemanticChunker(self.embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)

    def process_pdf(self, file_path):
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        documents = self.text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, self.embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        full_text = "\n".join([doc.page_content for doc in documents])
        return retriever, full_text

    def process_csv(self, file_content):
        df = pd.read_csv(file_content)
        summary = f"Columns: {', '.join(df.columns)}\nSample Data (first 5 rows):\n{df.head().to_string()}"
        full_data = df.to_string()
        return df, summary, full_data

    def process_image(self, image, question):
        image_base64 = self._img_to_base64(image)
        response = ollama.chat(
            model=self.llm.model,
            messages=[{"role": "user", "content": question, "images": [image_base64]}]
        )["message"]["content"]
        return self._clean_response(response)

    def _img_to_base64(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _clean_response(self, response):
        """Remove <think> tags and variations, then strip whitespace."""
        # Use regex to catch <think> and </think> with any attributes or spacing
        cleaned = re.sub(r'<think.*?>.*?</think>|<think.*?>', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def answer_question(self, question, retriever=None, csv_summary=None, csv_data=None, full_text=None):
        is_summary_request = "summar" in question.lower() and "file" in question.lower() or "about" in question.lower()

        if is_summary_request:
            if csv_data:
                context = f"{csv_summary}\nFull data: {csv_data}"
            elif full_text:
                context = full_text
            else:
                return "No file uploaded to describe."
        else:
            if csv_data:
                context = f"{csv_summary}\nFull data: {csv_data}"
            elif retriever:
                qa_chain = RetrievalQA(
                    combine_documents_chain=StuffDocumentsChain(
                        llm_chain=LLMChain(llm=self.llm, prompt=UNIVERSAL_PROMPT),
                        document_variable_name="context"
                    ),
                    retriever=retriever
                )
                response = qa_chain(question)["result"]
                return self._clean_response(response)
            else:
                context = ""

        if not context and not is_summary_request:
            response = self.llm.invoke(question)
        else:
            formatted_prompt = UNIVERSAL_PROMPT.format(context=context, question=question)
            response = self.llm.invoke(formatted_prompt)
        
        return self._clean_response(response)

def get_available_models():
    models_info = ollama.list()
    return tuple(sorted(model.model for model in models_info["models"]))