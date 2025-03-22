import os
import fitz #for pdf loading
from flask import Flask, render_template, request #for web app and to render html
from sentence_transformers import SentenceTransformer #for sentence embedding
from werkzeug.utils import secure_filename #It is from flask to secure and upload files in the backend server
from sklearn.metrics.pairwise import cosine_similarity #for cosine similarity

app = Flask(__name__) #creating and Initiating flask web app
app.config['UPLOAD_FOLDER'] = 'uploads' #setting upload folder I have created in the same directory in the backend
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) #creating the folder if it does not exist
#the above line is to create a folder named uploads in the same directory where the app.py is present
#It won't create if it already exists and it won't create a new folder when the app is run each time

model = SentenceTransformer('all-MiniLM-L6-v2') #loading the sentence transformer model

def extract_text_from_pdf(pdf_path):#function to extract text from pdf
    text=""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = text + page.get_text()
    return text.strip()

def cosines_similarity(resume_text, job_desc):#function to calculate cosine similarity resume text is from previous function and job_desc is from the form createted in the web app using html and flask
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0]*100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if "resume" not in request.files:
            return "No file uploaded", 400
        file = request.files['resume']
        job_desc = request.form['job_desc']

        if file.filename == "" or job_desc == "":
            return "invalid input", 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score = cosines_similarity(resume_text=resume_text, job_desc=job_desc)

        return render_template('index.html', score=score)
    return render_template('index.html', score=None)
    

if __name__ =="__main__":
    app.run(debug=True) #running the app in debug mode
    


