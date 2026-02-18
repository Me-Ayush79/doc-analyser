from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


class ResumeAnalysis(BaseModel):
    candidate_name: str = Field(
        description="Full name of the candidate. If not found, return 'not specified'."
    )
    email: str = Field(
        description="Email address. If not found, return 'not specified'."
    )
    phone: str = Field(
        description="Phone number. If not found, return 'not specified'."
    )
    total_experience_years: str = Field(
        description="Total professional experience in years. If not mentioned, return 'not specified'."
    )

    # Extracted Information
    skills: List[str] = Field(
        description="List of technical and soft skills. If none found, return ['not specified']."
    )
    education: List[str] = Field(
        description="Educational qualifications. If none found, return ['not specified']."
    )
    certifications: List[str] = Field(
        description="Certifications. If none found, return ['not specified']."
    )
    projects: List[str] = Field(
        description="Notable projects. If none found, return ['not specified']."
    )

    # Evaluation Metrics
    skill_relevance_score: float = Field(
        description="Score between 0-10 based on skill relevance."
    )
    experience_score: float = Field(
        description="Score between 0-10 based on experience quality."
    )
    project_score: float = Field(
        description="Score between 0-10 based on project quality."
    )
    overall_score: float = Field(
        description="Average of skill_relevance_score, experience_score and project_score."
    )

    # Final Verdict
    final_verdict: str = Field(
        description="Return 'valuable' if overall_score >= 6 else 'not_valuable'."
    )


output_parser = JsonOutputParser(pydantic_object=ResumeAnalysis)
format_instructions = output_parser.get_format_instructions()


"""
steps
loads resume 
uses jsonoutput parser to check resume, returns basic details

final verdict based on specified parameters: valuable or not
"""

parser = JsonOutputParser(pydantic_object=ResumeAnalysis)

prompt = PromptTemplate(
    template="""
You are an expert resume evaluator.

Instructions:

1. Extract structured information from the resume.
2. If ANY field is missing in the resume:
   - For string fields → return "not specified"
   - For list fields → return ["not specified"]
3. Score the resume using:
   - Skill relevance (0-10)
   - Experience quality (0-10)
   - Project strength (0-10)
4. Calculate overall_score as the average of the three scores.
5. If overall_score >= 6 → final_verdict = "valuable"
   Else → final_verdict = "not_valuable"

Return ONLY valid JSON.
Do not include explanations.

{format_instructions}

Resume:
{resume_text}
""",
    input_variables=["resume_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def init_llm_call(request: str, model, **kwargs):
    llm = ChatGoogleGenerativeAI(model=model, **kwargs)
    chain = prompt | llm | parser
    return chain.invoke({"resume_text": request})


class Analyser:
    def __init__(self, pdf, model: str, **model_params):
        self.pdf = pdf
        self.model = model
        self.model_params = model_params

    def load_resume(self):
        loader = PyPDFLoader(self.pdf)
        resume = loader.load()
        resume_content = ""

        for i in range(len(resume)):
            resume_content += resume[i].page_content

        return resume_content

    def Analyse_resume(self, content):
        analysis = init_llm_call(request=content, model=self.model, **self.model_params)
        return analysis
