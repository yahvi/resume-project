from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load ML model (only once)
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route("/rank", methods=["POST"])
def rank():
    try:
        data = request.json

        job_desc = data.get("job_desc", "")
        resumes = data.get("resumes", [])

        # Basic validation
        if not job_desc or not resumes:
            return jsonify({"error": "Missing job description or resumes"}), 400

        # Convert text to embeddings
        job_embedding = model.encode([job_desc])
        resume_embeddings = model.encode(resumes)

        # Compute similarity scores
        scores = cosine_similarity(job_embedding, resume_embeddings)[0].tolist()

        return jsonify({"scores": scores})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
