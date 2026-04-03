import streamlit as st
import requests

st.title("AI Resume Screening System")

# Job description input
job_desc = st.text_area("Enter Job Description")

# Upload resumes
uploaded_files = st.file_uploader(
    "Upload Resume Files (.txt only)",
    type=["txt"],
    accept_multiple_files=True
)

# Read resume content
resumes = []
for file in uploaded_files:
    content = file.read().decode("utf-8")
    resumes.append(content)

# Button
if st.button("Rank Candidates"):

    if not job_desc or not uploaded_files:
        st.warning("Please provide job description and upload resumes")

    else:
        try:
            # Send data to backend
            response = requests.post(
                "http://127.0.0.1:5000/rank",
                json={
                    "job_desc": job_desc,
                    "resumes": resumes
                }
            )

            # Get scores
            scores = response.json()["scores"]

            # Convert to percentage and rank
            ranked = [(i+1, score*100) for i, score in enumerate(scores)]
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

            # Show top candidate
            top_resume, top_score = ranked[0]
            st.success(f"🏆 Top Candidate: Resume {top_resume} ({top_score:.0f}%)")

            # Show ranking
            st.subheader("Ranking Results")
            for rank, (resume_num, score) in enumerate(ranked, start=1):
                st.write(f"Rank {rank}: Resume {resume_num} ({score:.0f}%)")

        except Exception as e:
            st.error(f"Error: {e}")
