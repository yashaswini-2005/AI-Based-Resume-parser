# AI-Powered-Resume-Parser

# Resume Parser with Job Title Classification

This project extracts key information (such as job title, name, email, phone number, education, and skills) from resumes in PDF format. It also classifies resumes based on predefined job titles using a machine learning model built with TensorFlow and Flask.

## Features

- *PDF Text Extraction*: Extracts text from PDF resumes using pdfplumber.
  
- *Text Preprocessing*: Cleans the extracted text, tokenizes, removes stopwords, and stems words.
  
- *Job Title Classification*: Classifies the resume into a job title based on keyword mapping using a trained deep learning model (Bidirectional LSTM with Conv1D).
  
- *Skills & Education Extraction*: Extracts key skills and education information from the resume.
  
- *File Upload & Parsing*: Upload resumes through the Flask web interface, or parse resumes stored in a directory.
  
- *Download Results*: Save the extracted information in CSV or JSON formats.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
  
- TensorFlow
  
- Keras
  
- Flask
  
- pdfplumber
  
- NLTK
  
- NumPy
  
- Pandas
  
- Scikit-learn
  
- phonenumbers

*Install the required Python packages using:*
```bash
pip install tensorflow keras flask pdfplumber nltk numpy pandas scikit-learn phonenumbers
```
## How to Run

1. *Clone this repository:*
```bash
https://github.com/Ghanasree-S/AI-Powered-Resume-Parser.git
```

2. *Navigate into the project directory:*
```bash
cd resume-parser
```

3. *Install dependencies:*
```bash
pip install -r requirements.txt
```

4. *Run the Flask app:*
```bash
python app.py
```

5. *Open your browser and go to http://127.0.0.1:5000/ to access the resume upload interface.*

## How It Works

1. *Text Extraction:* Uploaded PDF resumes are processed using pdfplumber to extract text.

2. *Preprocessing:* The extracted text is cleaned, tokenized, and processed by removing stopwords.
   
3. *Job Title Classification:* The processed text is passed through a deep learning model (Bidirectional LSTM with Conv1D) to predict the most likely job title.
   
4. *Information Extraction:* Using regular expressions and keyword matching, the system extracts key information such as the candidate’s name, email, phone number, skills, and education.
   
5. *Results Display:* The parsed results are shown on the web interface and can be downloaded as CSV or JSON files.
   
## API Endpoints

1. / - Home page for uploading resumes and displaying parsed results.
 
2. /upload - Endpoint for uploading a single resume for parsing.
   
3. /parse - Endpoint for parsing resumes stored in the uploads/ directory.
   
4. /download/csv - Download the parsed resume data as a CSV file.
   
5. /download/json - Download the parsed resume data as a JSON file.
   
## Example Usage

1. Upload a resume in PDF format through the web interface.
   
2. View the extracted information (name, email, phone number, job title, skills, education) and job title prediction.
   
3. Download the parsed data as a CSV or JSON file for further use or analysis.
   
## Dataset

The job title classification model was trained on a custom dataset containing resumes labeled with various job titles, such as "Software Engineer", "Data Analyst", "Project Manager", and more. This dataset was used to fine-tune a deep learning model to ensure high accuracy in job title prediction.

## Conclusion
This project allows for efficient resume parsing and classification, simplifying the recruitment process by automating the extraction of key information and classifying candidates based on their likely job titles. It provides a user-friendly interface for uploading resumes and offers the ability to download parsed results in multiple formats.
