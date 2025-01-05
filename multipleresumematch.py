__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool, SerperDevTool
import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Initialize tools
pdftool = PDFSearchTool()
serper_tool = SerperDevTool()
try:
    serper_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
    tools=[serper_tool]
    st.success("SerperDevTool initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize SerperDevTool: {e}")

def create_agents():
    parser_agent = Agent(
        role="Resume Parser",
        goal='Parse resume content and extract key information',
        backstory='Expert technical recruiter specializing in resume analysis',
        tools=[pdftool],
        verbose=True
    )

    job_searcher_agent = Agent(
        role="Job Searcher",
        goal='Search and analyze job postings from various websites based on the job search keywords',
        backstory='Specialized in finding relevant job opportunities',
        tools=[serper_tool],
        verbose=True
    )

    matcher_agent = Agent(
        role="Resume Matcher",
        goal='Match resume against job requirements',
        backstory='Expert in evaluating candidate-job fit',
        verbose=True
    )

    summarizer_agent = Agent(
        role="Summarizer",
        goal='Summarize matches and provide recommendations',
        backstory='Experienced in providing career guidance',
        verbose=True
    )
    
    return parser_agent, job_searcher_agent, matcher_agent, summarizer_agent

def search_jobs(resume_text: str, job_keywords: str, job_searcher_agent: Agent) -> list:
    search_task = Task(
        description=f'''
        Search for relevant jobs using these keywords: {job_keywords}
        Find jobs matching these skills from resume: {resume_text}
        Return top 5 most relevant positions with:
        1. Job title
        2. Company
        3. Location
        4. Key requirements
        5. Application link
        ''',
        expected_output="List of top 5 job postings with details",
        agent=job_searcher_agent
    )
    return search_task

def analyze_single_resume(file_path: str, job_keywords: str, agents):
    try:
        parser_agent, job_searcher_agent, matcher_agent, summarizer_agent = agents
        
        # Extract resume text
        with pdfplumber.open(file_path) as pdf:
            resume_text = "\n".join(page.extract_text() for page in pdf.pages)

        # Create tasks
        parsing_task = Task(
            description=f"Parse resume and extract key information: {resume_text}",
            expected_output="Structured resume information with skills, experience, and qualifications",
            agent=parser_agent
        )

        job_search_task = search_jobs(resume_text, job_keywords,job_searcher_agent)

        matching_task = Task(
            description="Compare resume against each job posting. Provide match analysis.",
            expected_output="Detailed match analysis for each job posting",
            agent=matcher_agent,
            context=[parsing_task, job_search_task]
        )

        summary_task = Task(
            description='''
            Provide for each job:
            1. Match percentage
            2. Key matching qualifications
            3. Skill gaps
            4. Application recommendations
            ''',
            expected_output="Summary of job matches with recommendations",
            agent=summarizer_agent,
            context=[parsing_task, job_search_task, matching_task]
        )

        crew = Crew(
            agents=[parser_agent, job_searcher_agent, matcher_agent, summarizer_agent],
            tasks=[parsing_task, job_search_task, matching_task, summary_task],
            verbose=True
        )

        return crew.kickoff()

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"

def process_multiple_resumes(files, job_keywords):
    results = []
    agents = create_agents()
    
    # Create a temporary directory for storing uploaded files
    if not os.path.exists('temp_resumes'):
        os.makedirs('temp_resumes')
    
    try:
        for file in files:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join('temp_resumes', file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process the resume
            result = analyze_single_resume(temp_file_path, job_keywords, agents)
            results.append({
                'filename': file.name,
                'analysis': result
            })
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
    finally:
        # Clean up the temporary directory
        if os.path.exists('temp_resumes'):
            os.rmdir('temp_resumes')
    
    return results

# Streamlit UI
st.header("Multiple Resume Job Matching System")

openai_api_key = st.text_input("OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please enter OpenAI API key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

job_keywords = st.text_input("Enter job search keywords (e.g., 'Python Developer New York'):")
resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Find Matching Jobs"):
    if resumes and job_keywords:
        st.info(f"Analyzing {len(resumes)} resumes and searching jobs...")
        
        # Process all resumes
        results = process_multiple_resumes(resumes, job_keywords)
        
        # Display results for each resume
        for result in results:
            st.subheader(f"Results for {result['filename']}")
            st.write(result['analysis'].raw)
            st.divider()
            
    else:
        st.warning("Please upload at least one resume and enter job keywords.")   