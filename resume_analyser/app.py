import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from resume_analyser import Analyser
import tempfile
import os


st.title("Resume Ranking System")

uploaded_files = st.file_uploader(
    "Upload resumes (PDF only)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:

    def process_resume(file):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        analyser = Analyser(temp_path, model="gemini-2.5-flash-lite", temperature=0.3)

        content = analyser.load_resume()
        result = analyser.Analyse_resume(content)

        os.remove(temp_path)  # Clean up temp file

        return file.name, result

    if st.button("Analyze Resumes"):
        with st.spinner("Analyzing resumes..."):
            with ThreadPoolExecutor(max_workers=3) as executor:
                results_list = list(executor.map(process_resume, uploaded_files))

        results = {name: analysis for name, analysis in results_list}

        highest_resume = max(results, key=lambda x: results[x]["overall_score"])

        st.subheader("Results")

        for name, result in results.items():
            st.write(f"**{name}** → Score: {result['overall_score']}")

        st.success(f"Highest: {highest_resume}")
        st.info(f"Verdict: {results[highest_resume]['final_verdict']}")
