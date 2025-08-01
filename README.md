# 🧠 LLM-Based Structured Data Extractor

A Python application that uses OpenAI's GPT-4 to extract structured JSON data from unstructured `.txt` files and stores the output in MongoDB. The app also uses Docker to spin up MongoDB and a web interface (Mongo Express) for easy data inspection.

---

## 📦 Features

- 📄 Reads unstructured text from `.txt` files
- 🧠 Uses GPT-4 (via OpenAI API) for intelligent data extraction
- ✂️ Automatically splits large text into manageable token-aware chunks
- 🛡 Robust error handling and logging
- 🧩 Combines partial outputs from multiple chunks
- 🗂 Stores results in MongoDB
- 🌐 Includes Mongo Express dashboard via Docker

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenAI GPT-4
- MongoDB
- Mongo Express
- Docker & Docker Compose
- LangChain (optional for vector DB/RAG support)

---

## 📁 Project Structure

├── app.py # Main application script
├── test.txt # Sample input text file
├── .env # Environment variables
├── docker-compose.yml # MongoDB + Mongo Express configuration
└── README.md # Project documentation


---

## 🔧 Setup Instructions

1. Clone the Repository
git clone https://github.com/yourusername/llm-data-extractor.git
cd llm-data-extractor

2. Create .env File
Create a .env file in the root folder and add your keys:

.env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# MongoDB config
MONGO_INITDB_ROOT_USERNAME=root
MONGO_INITDB_ROOT_PASSWORD=example

# Mongo Express (optional)
ME_CONFIG_BASICAUTH=false

3. Start MongoDB & Mongo Express
   docker-compose up -d
  
4. Prepare Your Input
Replace test.txt with any .txt file containing unstructured content you want to extract data from.

6. Run the Script
   python app.py
   The script will:

 Detect file encoding

 Split text into chunks (based on tokens)

 Extract structured data using GPT-4

 Combine partial results

 Insert JSON into MongoDB

 
📋## Data Extracted
You can customize the extraction template inside app.py. The current version extracts:

1. Contact Information
Contact number

Email ID

Address

2. Attachments
Media URLs

Media type

3. Social Profiles
LinkedIn, Twitter, Facebook, Instagram, YouTube, Other URLs

4. Team Member Information
Name, Designation, LinkedIn, USP

5. Ratings & Awards
Platform name, Ratings, Award images

6. Compliance
Certification, Issuer, Validity

7. Portfolio
Category, Price range, Projects completed, Client size

8. Financial Documents
KYC files, Document numbers

📈 View Results
Use Mongo Express to view your extracted results:

📍 URL: http://localhost:8081
🗂 DB: scraped_data
📄 Collection: extracted_info

📌 Future Enhancements
 Add PDF and DOCX file support

 Add semantic search using FAISS + LangChain

 Build web UI for file upload and preview

 Add Anki flashcard and CSV export support

 Enable retry with exponential backoff for OpenAI API

🤝 Contributing
Contributions are welcome!
Please open an issue or pull request with improvements or feature suggestions.

🛡 License
This project is licensed under the MIT License.
Feel free to use it in personal or commercial projects.

👨‍💻 Author
Raghav kr
GitHub: @Pomp007
LinkedIn: https://www.linkedin.com/in/shivam-kumar-660501288/

⭐️ Show Your Support
If you find this project helpful, please consider giving it a ⭐️ on GitHub!








