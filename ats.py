import os
import pymupdf
from dotenv import load_dotenv
from groq import Groq
from datetime import date

#pydantic class

from pydantic import BaseModel, Field

class WorkExperience(BaseModel):
    company: str
    job_title: str
    start_date: str
    end_date: str

class ResumeData(BaseModel):
    skills: list[str]
    education: list[str]
    work_experience: list[WorkExperience]

class ResumeScore(BaseModel):
    formatting: int = Field(ge=0, le=20)
    skills: int = Field(ge=0, le=20)
    experience: int = Field(ge=0, le=25)
    achievements: int = Field(ge=0, le=20)
    completeness: int = Field(ge=0, le=15)

class ResumeRecommendations(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]


#load api key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(bool(api_key))

#load and extract resume
pdf_path = "resume.pdf"
with pymupdf.open(pdf_path) as pdf:
    resume_text = ""

    for page in pdf:
        resume_text += page.get_text()
# print(resume_text[:200])
if not resume_text.strip():
    raise ValueError("No text found in the PDF.")

#llm call

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert resume information extractor. "
                "Extract skills, education, and work experience from the resume. "
                "Return only valid JSON with these exact keys: "
                "skills, education, work_experience. "
                "Skills and education must be lists of strings. "
                "Work_experience must be a list of objects. "
                "Each object must contain company, job_title, "
                "start_date, and end_date as strings. "
                "Format dates as YYYY-MM when possible, "
                "or YYYY if only the year is given. "
                "Use Present for current employment. "
                "Do not invent missing information. "
                "Use empty strings for missing fields and "
                "empty lists for missing sections."
            )
        },
        {
            "role": "user",
            "content": f"Extract information from this resume:\n\n{resume_text}"
        }
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

raw_output = response.choices[0].message.content

resume_data = ResumeData.model_validate_json(raw_output)

print(resume_data.model_dump_json(indent=2))

# print(response.choices[0].message.content)
#calculate year of exp
def calculate_experience(work_experience):
    months = set()

    for job in work_experience:
        start = job.start_date
        end = job.end_date

        if end.lower() == "present":
            end = date.today().strftime("%Y-%m")

        if len(start) != 7 or len(end) != 7:
            continue

        start_year, start_month = map(int, start.split("-"))
        end_year, end_month = map(int, end.split("-"))

        first = start_year * 12 + start_month
        last = end_year * 12 + end_month

        if 1 <= start_month <= 12 and 1 <= end_month <= 12:
            months.update(range(first, last + 1))

    return round(len(months) / 12, 2)

total_experience = calculate_experience(resume_data.work_experience)

print("Total Years of Experience:", total_experience)

score_response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert resume evaluator. "
                "Evaluate the resume using these five categories: "
                "formatting (0-20), skills (0-20), "
                "experience (0-25), achievements (0-20), "
                "and completeness (0-15). "
                "Evaluate only information present in the resume. "
                "Formatting means text structure and organization, "
                "not visual appearance. "
                "Return only valid JSON containing exactly these "
                "five keys with integer scores. "
                "Do not invent qualifications or achievements."
            )
        },
        {
            "role": "user",
            "content": f"Evaluate this resume:\n\n{resume_text}"
        }
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

score_data = ResumeScore.model_validate_json(
    score_response.choices[0].message.content
)

print(score_data.model_dump_json(indent=2))

total_score = sum([
    score_data.formatting,
    score_data.skills,
    score_data.experience,
    score_data.achievements,
    score_data.completeness
])

print(f"Final Resume Score: {total_score}/100")


feedback_response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert resume reviewer. "
                "Analyze the resume and its evaluation scores. "
                "Identify strengths, weaknesses, and provide exactly "
                "5 actionable recommendations to improve resume quality. "
                "Return valid JSON with three keys: "
                "strengths, weaknesses, recommendations. "
                "Each value must be a list of strings. "
                "Do not invent experience, achievements, or qualifications."
            )
        },
        {
            "role": "user",
            "content": (
                f"Resume:\n{resume_text}\n\n"
                f"Evaluation scores:\n{score_data.model_dump_json()}\n"
                f"Overall score: {total_score}/100"
            )
        }
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

feedback = ResumeRecommendations.model_validate_json(
    feedback_response.choices[0].message.content
)

print(feedback.model_dump_json(indent=2))

import json

final_report = {
    "overall_score": total_score,
    "total_experience_years": total_experience,
    "skills": resume_data.skills,
    "education": resume_data.education,
    "work_experience": [
        job.model_dump()
        for job in resume_data.work_experience
    ],
    "score_breakdown": score_data.model_dump(),
    "strengths": feedback.strengths,
    "weaknesses": feedback.weaknesses,
    "recommendations": feedback.recommendations
}

print(json.dumps(final_report, indent=2))

from pathlib import Path

output_path = Path(__file__).parent / "resume_report.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(final_report, file, indent=2)

print(f"Report saved to: {output_path}")