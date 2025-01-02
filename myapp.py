__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from  crewai import Agent,Task,Crew,Process
from langchain_openai import ChatOpenAI
import spacy
import PyPDF2
import pdfplumber
import docx
import re
from typing import List,Dict
import streamlit as st
import os
from crewai_tools import PDFSearchTool
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

from dotenv import load_dotenv
load_dotenv()



#LOAD OPENAI KEY
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY='Your API Key'
pdftool=PDFSearchTool()
#Add a sample job desription

job_desc="""We are seeking a highly skilled and experienced Agile Coach to join our team.
The Agile Coach will be responsible for leading and facilitating Agile transformations across the organization, working closely with stakeholders
to identify and address process improvements, and developing and implementing training programs to enhance the skills of team members.
The candidate should have below attributes:
Skills- Scrum, Kanban, Spotify, Facilitation, OKR's
Work Experience- 10+ years of work experience as an Agile Coach
Education- Bachelor's degree in Computer Science or related field
Certifications- CSM or PSM, SAFe SPC or others
"""

#Create the resume parser agent
parser_agent=Agent(
    role="Resume Parser",
goal='Search the relevant information in the uploaded resume pdf file',
    backstory='You are an experienced technical recruiter',
    verbose=True,
    tools=[pdftool],
    allow_delegation=True
)
# Create the matcher_agent
matcher_agent = Agent(
    role="Resume Matcher",
    goal='Compare resume content against job requirements and provide detailed matching analysis',
    backstory='''You are an expert recruiter specialized in matching candidate profiles with job requirements.
    You analyze skills, experience, education, and certifications to determine the fit.''',
    verbose=True,
    allow_delegation=True
)
#Create the summarizer Agent
Summarizer_agent= Agent(
    role="Summarizer",
    goal='Generate the summary of the pdf file and compare the summary with the {job_description}',
    backstory='You are an experienced technical recruiter',
    verbose=True,
    allow_delegation=False
)

#Create the parser task
def resume_parser(file_path:str)->str:
    extracted_text=""
    try:
        #with open(file_path,'rb') as f:
        with pdfplumber.open(file_path) as pdf_reader:
        #pdf_reader=PyPDF2.PdfReader(f)
        #text=''
            for page in pdf_reader.pages:
                extracted_text+=page.extract_text() + "\n"
        if not extracted_text.strip():
            raise ValueError("No text extracted from the PDF. Please check the file format.")

        # Log the extracted text for troubleshooting
        print("Extracted Text:\n", extracted_text)

        #Create parser task
        parsing_task = Task(
            description=f'''
            Parse the resume and extract the following key information:
            1. Skills and technologies
            2. Work experience and duration
            3. Education qualifications
            4. Certifications
            5. Any other relevant information
            
            Resume text: {extracted_text}
            ''',
            expected_output='Structured information extracted from the resume',
            agent=parser_agent
        )

        #create matching task
        matching_task = Task(
            description=f'''
            Compare the parsed resume information against the following job requirements:
            
            Job Description:
            {job_desc}
            
            Provide detailed analysis of:
            1. Skills match and gaps
            2. Experience match (including years)
            3. Education match
            4. Certification match
            5. Overall suitability
            ''',
            expected_output='Detailed matching analysis between resume and job requirements',
            agent=matcher_agent,
            context=[parsing_task]
        )
        #create summary task
        summary_task = Task(
            description='''
            Generate a comprehensive summary including:
            1. Overall match percentage
            2. Key matching points
            3. Notable gaps or missing requirements
            4. Specific recommendations
            5. Final suitability assessment (Fit/Partial Fit/Not Fit)
            ''',
            expected_output='Clear summary with match analysis and recommendations',
            agent=Summarizer_agent,
            context=[parsing_task,matching_task]
        )
    #Define the crew
        crew= Crew(
        agents=[parser_agent,matcher_agent,Summarizer_agent],
        tasks=[parsing_task,matching_task,summary_task],
        verbose=True
        )
        result=crew.kickoff()
        return result.raw

    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred while processing the resume."

#Create a streamlit app to upload resume in pdf format and get the matching response
st.header("Find the resume match")
st.write("Upload your resume in pdf format")
st.write("The system will match your resume with the job description and provide a summary of the match")
resume=st.file_uploader("Upload your resume",type="pdf")
if st.button("Submit"):
    if resume is not None:
    # Save the uploaded file temporarily
        with open("uploaded_resume.pdf", "wb") as f:
            f.write(resume.getbuffer())

    # Call the resume parser function
        st.info("Analyzing resume.....please wait")
        response = resume_parser("uploaded_resume.pdf")
        st.subheader("Analysis Results")
        st.write(response)
else:
    st.write("Please upload a resume file")
    
#response=resume_parser()



         






