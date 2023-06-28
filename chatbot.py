# Install the required dependencies before deploying app on Streamlit
import os
import subprocess

# Install required packages
required_packages = [
    "streamlit",
    "pdfplumber",
    "panel",
    "textwrap"
]

for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


import streamlit as st
import openai
import pdfplumber
import panel as pn
import textwrap

# Set OpenAI API key
# openai.api_key = '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'

openai.api_key = os.environ.get("OPENAI_API_KEY")


# Helper function to get chatbot response

def get_chatbot_response(user_input):
    messages = [{'role': 'system', 'content': user_input}]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message['content']


# Helper function to process uploaded PDFs
def process_pdfs(file_uploads):
    cv_contents = []

    for file_upload in file_uploads:
        with pdfplumber.open(file_upload) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            cv_contents.append(text)

    return cv_contents

# Helper function for comparing CVs and making recommendations
def compare_cvs_and_recommend(cv_contents):
    # Job specifications for each vacancy
    job_specifications = {
        'QA Engineer – Automation': {
            'experience': '1 to 2 years of related experience in the software engineering or relevant field.',
            'automation_experience': 'At least 1 year of experience in Automation testing.',
            'tools': ['Selenium', 'Appium', 'Reset Assured'],
            'programming_language': 'Sound knowledge of Java programming language.',
            'degree': 'Bachelors’ Degree in Software Engineering, Computer Science, or an equivalent qualification.',
            'responsibilities': [
                'Reviewing quality specifications and technical design documents to provide timely and meaningful feedback.',
                'Creating detailed, comprehensive, and well-structured test plans and test cases.',
                'Estimating, prioritizing, planning, and coordinating quality testing activities.',
                'Identify, record, document thoroughly, and track bugs.',
                'Perform thorough regression testing when bugs are resolved.'
            ]
        },
        'Senior Java Developer': {
            'experience': 'Minimum of 3 years of related work experience.',
            'communication_skills': 'Excellent written & Verbal Communication skills.',
            'skills': [
                'Java / Spring Boot.',
                'Event-driven architecture, Micro Services.',
                'AWS.',
                'MongoDB, MySQL, Docker, Kafka.',
                'Design patterns, SOLID Principle.',
                'JIRA, Confluence.'
            ]
        },
        'AI Intern': {
            'skills': ['Machine learning', 'python', 'R', 'Deep learning'],
            'experience': 'No work experience needed.',
            'degree': 'Undergraduate or degree in Information Technology, specializing in Data Science.'
        }
    }

    recommendations = []

    for cv_content in cv_contents:
        # Compare CV with job specifications
        matches = {}

        for job_title, job_spec in job_specifications.items():
            match_count = 0

            if 'experience' in job_spec:
                if job_spec['experience'] in cv_content:
                    match_count += 1

            if 'skills' in job_spec:
                for skill in job_spec['skills']:
                    if skill in cv_content:
                        match_count += 1

            if 'degree' in job_spec:
                if job_spec['degree'] in cv_content:
                    match_count += 1

            matches[job_title] = match_count

        # Get recommended job position
        recommended_position = max(matches, key=matches.get)

        # Prepare recommendation text
        recommendation_text = f"Applicant:\n{cv_content}\nRecommended Position:\n{recommended_position}"

        recommendations.append(recommendation_text)

    return recommendations

# Streamlit app code
def main():
    st.title('Chatbot App')

    # User input
    user_input = st.text_input('User Input')

    # Send button
    if st.button('Send'):
        if user_input:
            # Get chatbot response
            chatbot_response = get_chatbot_response(user_input)

            # Display chatbot response
            st.text_area('Chatbot Response', value=chatbot_response, height=200)

    # File upload
    uploaded_files = st.file_uploader('Upload CV(s)', type='pdf', accept_multiple_files=True)

    # Process uploaded files and display
    if uploaded_files:
        cv_contents = process_pdfs(uploaded_files)
        recommendations = compare_cvs_and_recommend(cv_contents)
        for i, recommendation in enumerate(recommendations):
            st.text_area(f'Recommendation {i+1}', value=recommendation, height=200)

if __name__ == '__main__':
    main()
