# RAG Chatbot: Retrieval-Augmented Generation with Ollama and Pinecone

## 🚀 Project Overview

This RAG (Retrieval-Augmented Generation) Chatbot is a sophisticated AI-powered document analysis and question-answering system that combines document embedding, vector storage, and language model generation. It leverages Ollama for embeddings and language models, Pinecone for vector storage, and LangChain for document processing.

## ✨ Key Features

- 📄 PDF Document Processing
- 🔍 Semantic Search and Retrieval
- 💬 Context-Aware Response Generation
- 🧠 Conversation Memory
- 🔒 Secure Environment Variable Management

## 🛠 Technologies Used

- **Language Model**: Ollama (Deepseek-r1:1.5b)
- **Embeddings**: Nomic Embed Text
- **Vector Database**: Pinecone
- **Document Processing**: LangChain
- **Programming Language**: Python

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running
- Pinecone account and API key
- Dependencies installed

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Kawaki-Nagisa/AI-RAG-Agent-with-Ollama-and-Pinecone.git
cd PineCone-RAG-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

## 🚀 Quick Start

### Configuration

You can customize the RAG Chatbot through various configuration parameters:

```python
config = RAGChatbotConfig(
    index_name="my-custom-index",
    embedding_model="nomic-embed-text",
    chunk_size=500,
    chunk_overlap=250
)
```

### Basic Usage

```python
# Initialize the RAG Chatbot
rag_chatbot = RAGChatbot(config)

# Process a document
rag_chatbot.process_document("your_document.pdf")

# Query the processed document
response = rag_chatbot.generate_response("Your query here")
print(response)
```

## 🛠 Configuration Options

### RAGChatbotConfig Parameters

- `pinecone_api_key`: Pinecone API key (optional, can be set via environment variable)
- `index_name`: Name of the Pinecone index (default: "test")
- `embedding_model`: Embedding model to use (default: "nomic-embed-text")
- `chunk_size`: Size of text chunks (default: 500)
- `chunk_overlap`: Overlap between text chunks (default: 250)

## 📦 Project Structure

```
rag-chatbot/
│
├── main.py                 # Main script
├── .env                    # Environment variables
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🔒 Security Considerations

- Always keep your Pinecone API key confidential
- Use environment variables for sensitive information
- Implement proper access controls

## 📈 Performance Optimization

- Adjust `chunk_size` and `chunk_overlap` for optimal document processing
- Choose appropriate embedding and language models
- Monitor vector store performance and index size

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Support

For issues, please open a GitHub issue or contact the maintainer.

## 🙏 Acknowledgements

- [Ollama](https://ollama.ai/)
- [Pinecone](https://www.pinecone.io/)
- [LangChain](https://www.langchain.com/)

---

**Note**: This project is a demonstration of RAG technology and should be adapted to specific use cases and production requirements.
