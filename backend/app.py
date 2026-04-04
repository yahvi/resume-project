from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

# ----------------------------
# Health Check Route
# ----------------------------
@app.route("/")
def home():
    return "Resume Backend is live and running!"

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# ----------------------------
# Ranking API
# ----------------------------
@app.route("/rank", methods=["POST"])
def rank():

    try:
        data = request.get_json()

        job_desc = data.get("job_description", "")
        resumes = data.get("resumes", [])

        # Validation
        if not job_desc or not resumes:
            return jsonify({"error": "Missing input"}), 400

        # Clean text
        job_desc = clean_text(job_desc)
        resumes = [clean_text(r) for r in resumes]

        # Combine text
        corpus = [job_desc] + resumes

        # TF-IDF with improvements
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Split vectors
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        # Cosine similarity
        scores = cosine_similarity(job_vector, resume_vectors)[0]

        # Normalize scores
        if scores.max() != 0:
            scores = scores / scores.max()

        # Rank resumes
        ranked = sorted(
            zip(resumes, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Prepare response
        result = []
        for i, (resume, score) in enumerate(ranked):
            result.append({
                "rank": i + 1,
                "resume": resume,
                "score": round(float(score) * 100, 2)
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# RUN (RENDER SAFE)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
