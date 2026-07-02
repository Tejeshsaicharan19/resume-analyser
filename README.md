# resume-analyser
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# App imports
from parser.resume_parser import ResumeParser
from parser.jd_parser import JDParser
from scoring.ats_engine import calculate_ats_score
from ml.predictor import predict_shortlisting
from rag.vector_store import process_document_to_store
from agents.supervisor import agent_app
from memory.memory_manager import MemoryManager
from dashboard.analytics import render_dashboard
from reports.pdf_generator import generate_pdf_report

load_dotenv()

st.set_page_config(page_title="Agentic Resume Analyzer", layout="wide", page_icon="📄")

def init_session():
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager(session_id="user_session")
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = None
    if "jd_data" not in st.session_state:
        st.session_state.jd_data = None
    if "ats_results" not in st.session_state:
        st.session_state.ats_results = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

init_session()

st.sidebar.title("AI Resume Analyzer 🤖")
st.sidebar.markdown("Upload your Resume and Job Description to get started.")

# Sidebar Uploads
st.sidebar.subheader("1. Upload Resume")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], key="resume")

st.sidebar.subheader("2. Job Description")
jd_file = st.sidebar.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd")
jd_text = st.sidebar.text_area("Or Paste JD Here", key="jd_text")

if st.sidebar.button("Analyze Resume", type="primary"):
    if not resume_file:
        st.sidebar.error("Please upload a resume first.")
    else:
        with st.spinner("Parsing documents..."):
            # Process Resume
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp:
                tmp.write(resume_file.read())
                resume_path = tmp.name
                
            parser = ResumeParser(resume_path)
            st.session_state.resume_data = parser.parse()
            os.remove(resume_path)
            
            # Process JD
            if jd_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp:
                    tmp.write(jd_file.read())
                    jd_path = tmp.name
                jd_parser = JDParser(jd_path, is_file=True)
                st.session_state.jd_data = jd_parser.parse()
                os.remove(jd_path)
            elif jd_text:
                jd_parser = JDParser(jd_text, is_file=False)
                st.session_state.jd_data = jd_parser.parse()
            else:
                st.session_state.jd_data = {"skills": [], "raw_text": ""}
                
            # Score and Predict
            st.session_state.ats_results = calculate_ats_score(st.session_state.resume_data, st.session_state.jd_data)
            
            features = {
                'ats_score': st.session_state.ats_results['overall_score'],
                'skills_count': len(st.session_state.resume_data.get('skills', [])),
                'experience_years': 3, # Placeholder, would extract from text
                'projects_count': 2, # Placeholder
                'certs_count': 1 # Placeholder
            }
            st.session_state.prediction = predict_shortlisting(features)
            
            # Update Vector DB
            combined_text = st.session_state.resume_data.get("raw_text", "") + "\n\n" + st.session_state.jd_data.get("raw_text", "")
            if combined_text.strip():
                try:
                    process_document_to_store(combined_text)
                    st.sidebar.success("Vector DB Updated for Chat!")
                except Exception as e:
                    st.sidebar.error(f"Failed to update embeddings: {e}. Check API Key.")
            
        st.sidebar.success("Analysis Complete!")

# Main Layout
tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "💬 AI Career Coach & Agent Chat"])

with tab1:
    if st.session_state.ats_results:
        render_dashboard(st.session_state.ats_results, st.session_state.prediction)
        
        st.subheader("Generate Report")
        if st.button("Download PDF Report"):
            report_path = generate_pdf_report(
                st.session_state.resume_data, 
                st.session_state.ats_results
            )
            with open(report_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Report",
                    data=pdf_file,
                    file_name="Resume_Report.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Upload documents and click 'Analyze Resume' to see results.")

with tab2:
    st.header("Chat with your AI Agents")
    st.markdown("Ask anything! Our LangGraph Supervisor will route your question to the best specialized agent (ATS, Skills, Improvement, or Career Coach).")
    
    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Chat Input
    user_query = st.chat_input("Ask: 'How can I improve my summary?' or 'What skills am I missing?'")
    
    if user_query:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    # Invoke LangGraph Agent
                    inputs = {
                        "messages": [("user", user_query)],
                        "resume_data": st.session_state.resume_data or {},
                        "jd_data": st.session_state.jd_data or {}
                    }
                    
                    result = agent_app.invoke(inputs)
                    response_text = result["messages"][-1].content
                    st.write(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}. Please ensure GEMINI_API_KEY is set correctly.")
