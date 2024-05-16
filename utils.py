from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Optional,Union,Any
from pydantic import BaseModel, Field
from format import OutputFormat, SkillsFormat
from jobspy import scrape_jobs
import json
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import csv
from langchain_core.tools import tool
import os
import pandas as pd



class Utils:
    def __init__(
        self,
        llm_name: str = "gpt-3.5-turbo-0125",
        evalutor_llm_name: str = "gpt-4-turbo",
        embedding_model: str = "text-embedding-3-small"
    ) -> None:
        self.openai_chat_model = ChatOpenAI(temperature=0.1, top_p=0.0001, model=llm_name)
        self.openai_evalutor_model = ChatOpenAI(temperature=0.1, top_p=0.0001, model=evalutor_llm_name)
        self.enc = tiktoken.encoding_for_model(llm_name)
        self.pages = PyPDFLoader("data/resume.pdf").load_and_split()
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)

    def get_resume(self, path):
        if os.path.isfile(path):
            with open(path) as json_file:
                print('Loading resume.json from the disk')
                resume_json= json.load(json_file)
        else:
            print('Reading pdf resume and Loading')
            resume_json = self.UtilsObject.read_parse_resume()
            self.UtilsObject.write_resume_to_file(resume_json)
        self.resume = resume_json
        self.summary = self.resume["About_Me"]["Summary"]
        self.keywords = self.resume["Skills"]["Technical_Skills"]

    def get_jobs(self, path):
        if os.path.isfile(path):
            print('Loading jobs.csv from the disk')
            csvFile = df=pd.read_csv(path)
        else:
            print('Scanning web for jobs and Loading')
            csvFile= self.UtilsObject.scan_for_jobs()
        self.jobs_dict= csvFile
        firstJob = self.jobs_dict.iloc[0]
        self.job_description = firstJob.description

    def read_parse_resume(self):
        PROMPT_4="""
            You are given resume : ```{resume}```

            then based on the given resume extract information about the person

            {format_instructions}

            """
        template = PROMPT_4
        parser = PydanticOutputParser(pydantic_object=OutputFormat)
        prompt_template_name = PromptTemplate(
            input_variables=["resume"],
            template=template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
            )

        name_chain = LLMChain(llm=self.openai_chat_model, prompt=prompt_template_name)
        response = name_chain(inputs={"resume": self.pages})
        resume_json=parser.parse(response["text"]).json()
        return resume_json
    
    def get_keywords_from_job_description(self, description):
        PROMPT_5="""
            #CONTEXT
            Read the job description and provide keywords of all data related skills, 
            data analysis tools and programming languages required!

            Format keywords as a List object like below. 
            ["skill 1", "skill 2", "skill 3"]
            
            Don't include any explanations in your responses
            
            No explanations, unique keywords, no duplicates. Be brief
            
            QUERY:
            {query}
            """
        chat_prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_5)
        ])
        skills_chain = chat_prompt | self.openai_chat_model ## add pydantic parser, 
        skills_list = skills_chain.invoke({"query": description})        
        # print(skills_list)
        return skills_list.content
    
    def adapt_resume_job_description(self, keywords, resume):
        SYSTEM_PROMPT = "You are an expert in Resume building given the list of technical skills mentioned in the job description"
        USER_PROMPT = """
            I need to adapt all my work responsibilities in my resume to use the technical skills mentioned in the job description. Here is my current resume:
            {resume}
            
            Here are the technical skills mentioned in the job description :
            {keywords}
            
            For each work responsibility, you MUST give the original and modified sentence. 
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        skills_chain = chat_prompt | self.openai_chat_model ## add pydantic parser, 
        skills_list = skills_chain.invoke({"resume": resume,  "keywords": keywords})        
        # print(skills_list)
        return skills_list.content

    def modify_summary(self, description, summary):
        SYSTEM_PROMPT = "You are an expert in resume building"
        USER_PROMPT = """
            I need to update my resume summary to suit the job description.  Here is my current summary:
            {summary}
            
            Here is the job description :
            {description}
            
            You MUST highlight the technical and leadership skills that are mentioned in the job description. Return the original summary along with and modified summarys. 
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        skills_chain = chat_prompt | self.openai_chat_model ## add pydantic parser, 
        skills_list = skills_chain.invoke({"description": description,  "summary": summary})        
        # print(skills_list)
        return skills_list.content

    @tool
    def missing_keywords_from_resume(self):
        """
        Given a set of keywords and a job description, finds all the missing keywords in the job description.
        
        Args:
            keywords: list of keywords
            description: job description
        """
        SYSTEM_PROMPT = "You are an expert in reading resume's and job description"
        USER_PROMPT = """
            You are given a job description. Here is the job description:
            {self.job_description}
            
            You are also given the list of user skillset :
            {self.keywords}
            
            List out all the skills which are in the job description but missing from the user skillset in the order of importance.
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        chain = chat_prompt | self.openai_chat_model ## add pydantic parser, 
        response = chain.invoke()        
        return response.content

    @tool
    def inconsistencies_in_resume(self):
        """
            Finds conflicting information, date inconsistencies in the given resume
            
            Args:
                resume: resume
        """
        SYSTEM_PROMPT = "You are an expert in reading resume's"
        USER_PROMPT = """
            You are given resume ```{self.resume}```. 

            List out all the conflicting information, date inconsistencies.
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        chain = chat_prompt | self.openai_chat_model ## add pydantic parser, 
        response = chain.invoke()        
        # print(skills_list)
        return response.content

    def scan_for_jobs(self):
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
            search_term="software engineer",
            location="San Francisco, CA",
            results_wanted=20,
            hours_old=72, # (only Linkedin/Indeed is hour specific, others round up to days old)
            country_indeed='USA',  # only needed for indeed / glassdoor
            # linkedin_fetch_description=True # get full description and direct job url for linkedin (slower)
        )
        print(f"Found {len(jobs)} jobs")
        jobs.to_csv("data/jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_xlsx
        return jobs

    def write_resume_to_file(self, resume_json, fileName:str='data/resume.json'): 
        f=open(fileName, 'w')
        f.write(resume_json)
        f.close()
    
    def evaluate_resume_job_description(self, resume, description):
        SYSTEM_PROMPT = "You are an expert matching a resume with the job description. You should be hyper-critical."
        USER_PROMPT = """
            You are given resume ```{resume}```. 
            You are also give the job description ```{description}```

            Provide the scores (out of 10) for the following attributes
            1. Overall match - How much the resume is a match to the job description
            1. Technical skill match - How many of the technical skills in the job description match the resume
            2. Communication skill match - Resume is a form of communication while persuing for a new job oportunity. Does the resume communicate clearly the summary and it's previous job responsibilities 
            
            Please take your time, and think through each item step-by-step, when you are done - please provide your response in the following JSON format:
            overall : "score_out_of_10",
            technical : "score_out_of_10",
            communication : "score_out_of_10"
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        skills_chain = chat_prompt | self.openai_evalutor_model ## add pydantic parser, 
        skills_list = skills_chain.invoke({"resume": resume, "description": description})        
        # print(skills_list)
        return skills_list.content

    #semanticTextSplitter
    #tokenRTextSplitter
    def split_into_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=self.tiktoken_len
        )
        self.split_chunks = text_splitter.split_documents(self.docs)
        
        # semantic splitter
        # text_splitter = SemanticChunker(OpenAIEmbeddings())
        # self.split_chunks = text_splitter.split_documents([self.docs])
        return self.split_chunks

    def get_llm_model(self):
        return self.openai_chat_model

    def init_prompt(self) -> ChatPromptTemplate:
        RAG_PROMPT = """
            ###Instruction###:
            Answer the question based only on the following context. If you cannot answer the question with the context, please respond with "I don't know":
            
            CONTEXT:
            {context}

            QUERY:
            {question}            
            """
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        return rag_prompt

    def tiktoken_len(self, text) -> int:
        self.tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
            text,
        )
        return len(self.tokens)

    def get_vector_store(self):
        self.qdrant_vectorstore = Qdrant.from_documents(
            self.split_chunks,
            self.embedding_model,
            location=":memory:",
            collection_name="meta-10k",
        )
        return self.qdrant_vectorstore
    
    def generate_test_set(self)-> None:
        text_splitter_eval = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 50
        )
        eval_documents = text_splitter_eval.split_documents(self.docs)
        distributions = {
            "simple": 0.5,
            "multi_context": 0.4,
            "reasoning": 0.1
        }
        testset = self.test_generator.generate_with_langchain_docs(eval_documents, 10, distributions, is_async = False)
        print("santhosh:"+len(testset.to_pandas()))
