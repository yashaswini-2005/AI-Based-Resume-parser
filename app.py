from flask import Flask, request, render_template, send_file, session
import os
import pdfplumber
import re
import phonenumbers
import csv
import json

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')

# Load intents from JSON
with open('intents.json') as json_file:
    intents_data = json.load(json_file)

# Create mapping from patterns to job labels
keyword_to_label = {}
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        keyword_to_label[pattern.lower()] = intent['tag']

# Skills list
skills_keywords = [
    "Python", "Machine Learning", "Data Analysis", "Project Management",
    "Java", "C++", "JavaScript", "SQL", "HTML", "CSS", "React",
    "Node.js", "Django", "Flask", "Excel", "PowerPoint", "Communication",
    "Teamwork", "Leadership", "Problem-Solving", "Agile", "Scrum"
]

def clean_text(text):
    return text.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return render_template('index.html', error="No file part.")

    file = request.files['resume']

    if file.filename == '':
        return render_template('index.html', error="No selected file.")

    if file and file.filename.endswith('.pdf'):
        upload_folder = 'uploads/'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(text)

        if not cleaned_text:
            return render_template('index.html', error="Failed to extract text from the uploaded PDF.")

        print(f"Extracted Text: {cleaned_text[:500]}...")

        # Extract details
        education = extract_education(cleaned_text)
        skills = extract_skills(cleaned_text)
        key_info = extract_key_information(cleaned_text)
        job_title = key_info['job_title']
        job_class = classify_job(skills, job_title, intents_data)

        results = [{
            'filename': file.filename,
            'name': key_info['name'],
            'email': key_info['email'],
            'phone': key_info['phone'],
            'job_title': job_title,
            'skills': skills,
            'education': education,
            'job_class': job_class
        }]

        session['results'] = results

        return render_template('index.html', results=results)
    else:
        return render_template('index.html', error="Invalid file format. Please upload a PDF file.")

def extract_job_title(text):
    text_lower = text.lower()
    for keyword, title in keyword_to_label.items():
        if keyword in text_lower:
            return title
    return "Job title not found."

def extract_key_information(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'(\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}'
    phone_matches = re.findall(phone_pattern, text)

    email = re.findall(email_pattern, text)
    name = text.split('\n')[0] if text else 'N/A'
    job_title = extract_job_title(text)

    normalized_phone = 'N/A'
    if phone_matches:
        for match in phone_matches:
            try:
                phone_number = phonenumbers.parse(match, "US")
                if phonenumbers.is_valid_number(phone_number):
                    normalized_phone = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)
                    break
            except phonenumbers.NumberParseException:
                continue

    return {
        'name': name,
        'email': email[0] if email else 'N/A',
        'phone': normalized_phone,
        'job_title': job_title
    }

def extract_skills(text):
    extracted_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return ", ".join(extracted_skills) if extracted_skills else "No skills found."

def extract_education(text):
    education_keywords = ["Bachelor", "Master", "PhD", "B.Sc", "M.Sc", "University", "College"]
    extracted_education = [edu for edu in education_keywords if edu.lower() in text.lower()]
    return ", ".join(extracted_education) if extracted_education else "No education information found."

def classify_job(parsed_skills, job_title, intents_data):
    parsed_skills = [skill.lower() for skill in parsed_skills.split(", ")]
    job_title = job_title.lower()

    matched_intent = None
    highest_score = 0

    for intent in intents_data['intents']:
        patterns = [p.lower() for p in intent['patterns']]
        match_score = 0

        for pattern in patterns:
            for item in parsed_skills + [job_title]:
                if pattern in item:
                    match_score += 1

        if match_score > highest_score:
            highest_score = match_score
            matched_intent = intent

    return matched_intent['responses'][0] if matched_intent else "Unknown Job Role"

@app.route('/download/csv', methods=['GET'])
def download_csv():
    results = session.get('results', [])
    output_file = 'parsed_resumes.csv'

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Name', 'Email', 'Phone', 'Job Title', 'Skills', 'Education', 'Job Classification'])
        for result in results:
            writer.writerow([
                result['filename'], result['name'], result['email'], result['phone'],
                result['job_title'], result['skills'], result['education'], result['job_class']
            ])

    return send_file(output_file, as_attachment=True)

@app.route('/download/json', methods=['GET'])
def download_json():
    results = session.get('results', [])
    output_file = 'parsed_resumes.json'

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
