from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/")
def home():
    return "Backend is running successfully!"

@app.route("/rank", methods=["POST"])
def rank():
    data = request.get_json()
    job_desc = data.get("job_desc", "")
    resumes = data.get("resumes", [])

    if not job_desc or not resumes:
        return jsonify({"error": "Missing inputs"}), 400

    docs = [job_desc] + resumes
    vectors = TfidfVectorizer().fit_transform(docs)

    scores = cosine_similarity(vectors[0], vectors[1:])[0]
    return jsonify({"scores": scores.tolist()})

if __name__ == "__main__":
    app.run()
