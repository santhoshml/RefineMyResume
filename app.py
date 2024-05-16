from dotenv import load_dotenv
load_dotenv() 

from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from utils import *
import os
import getpass
from langchain.globals import set_debug
import chainlit as cl 
from langchain_openai import ChatOpenAI, OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import csv
import json
import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Any, Callable, List, Optional, TypedDict, Union, Annotated

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
import functools
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
import functools
from langgraph.prebuilt import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

def write_resume_to_file(resume_json, fileName:str='data/resume.json'): 
    f=open(fileName, 'w')
    f.write(resume_json)
    f.close()

def read_parse_resume():
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
    print("santhosh",pages)
    name_chain = LLMChain(llm=openai_chat_model, prompt=prompt_template_name)
    response = name_chain(inputs={"resume": pages})
    resume_json=parser.parse(response["text"]).json()
    return resume_json

def get_resume(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            print('Loading resume.json from the disk')
            resume_json= json.load(json_file)
    else:
        print('Reading pdf resume and Loading')
        resume_json = read_parse_resume()
        write_resume_to_file(resume_json)
    return resume_json

def scan_for_jobs():
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

def get_jobs(path):
    if os.path.isfile(path):
        print('Loading jobs.csv from the disk')
        csvFile = df=pd.read_csv(path)
    else:
        print('Scanning web for jobs and Loading')
        csvFile= scan_for_jobs()
    return csvFile

@tool
def adapt_resume_job_description():
    """
        Given the list of skills in the job description and the resume. You can adapt/modify the initial resume to suit the job requirments.        
    """
    SYSTEM_PROMPT = "You are an expert in Resume building given the list of skills in the job description and the resume"
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
    chain = chat_prompt | openai_chat_model ## add pydantic parser, 
    return chain.invoke({"resume": resume,  "keywords": keywords})
    
@tool
def missing_keywords():
    """
        Use this method to find the missing skills from the resume for a given job description.
    """
    SYSTEM_PROMPT = "You are an expert in reading and analyzing a job description. You are expert in reading applican't resume for a job. You always respond in a kind way."
    USER_PROMPT = """
            # CONTEXT #
            I want to share the job description of a role required by our company. Also the resume of one of the applicant.
            
            Here is the job description:
            {job_description}
            
            Here is the  resume of the applicant:
            {resume}
            
            # OBJECTIVE #
            Compare the given Resume with the job description and List all the missing talents/skills/keywords from the resume.
            
            # STYLE #
            Be constructive but confident. Maintain neutral tone.
            
            # AUDIENCE #
            Tailor the reponse to the potential applicant, a software developer.
            
            # RESPONSE #
            Be concise and succinct in your response.
            Return List of all the talents/skills/keywords which are in the job description but missing from the resume.            
        """
    chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)                
        ])
    
    chain = chat_prompt | openai_chat_model ## add pydantic parser, 
    return chain.invoke({"job_description": job_description, "resume": resume})

@tool
def inconsistencies_in_resume():
    """
        Use this method to find discrepancies in the given resume.
    """
    SYSTEM_PROMPT = "You are an expert in reading and analyzing resume. You always respond in a kind way."
    USER_PROMPT = """
        # CONTEXT #
        Resume capture's the years of work experience, highlight relevant technical and non-technical skills.
        Here is the resume: 
        {resume}

        # OBJECTIVE #
        Find and list all the discrepancies in the given Resume.
        
        # STYLE #
        Be constructive but confident. Maintain neutral tone. Be hyper-critical.
        
        # AUDIENCE #
        Tailor the reponse to the person looking for a software developer job.
        
        # RESPONSE #
        Be concise and succinct in your response.
        Return List of all discrepancies.
    """
    chat_prompt = ChatPromptTemplate.from_messages(
        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT)
    ])
    chain = chat_prompt | openai_chat_model ## add pydantic parser, 
    return chain.invoke({"resume": resume})

@tool
def modify_summary():
        """
            use this method to find modify or update the summary in the resume.
        """
        SYSTEM_PROMPT = """
        You are an adept at interpreting job description and making sense of it.
        You are a master at modifying resume summaries. You always respond in a kind way.
        """
        USER_PROMPT = """
            # CONTEXT #
            A resume summary is a brief, concise paragraph. It serves as an introduction to the job seeker's career profile. 
            It includes a quick overview of the job seeker's most relevant qualifications, skills, experience, and sometimes notable achievements.
            Here is the initial summary:
            {summary}
            
            Here is the job description :
            {job_description}
            
            # OBJECTIVE #
            Read and analyze the job description. Modify or update the initial resume summary with respect to the job description.
            
            # AUDIENCE #
            Tailor the reponse to the potential applicant, a software developer.
            
            # STYLE #
            Be constructive and confident. Maintain optimistic tone.
            
            # RESPONSE #
            Be concise and succinct in your response.
            Return the summary that highlights the technical and leadership skills that are mentioned in the job description.
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", USER_PROMPT)
        ])
        chain = chat_prompt | openai_chat_model ## add pydantic parser, 
        return chain.invoke({"job_description": job_description,  "summary": summary})        

@tool
def evaluate_resume_job_description():
    """
        You are an expert in analysing resume's and the job requirments.
        You can rate how well a resumes matches up with the job description.
    """
    SYSTEM_PROMPT = "Given a resume and a job description evaluate there compatablity. You should be hyper-critical."
    USER_PROMPT = """
        You are given resume ```{resume}```. 
        You are also give the job description ```{job_description}```

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
    chain = chat_prompt | openai_evalutor_model ## add pydantic parser, 
    return chain.invoke({"resume": resume, "job_description": job_description})        

@tool
def check_for_job_description_related_quires():
    """
        You are an expert in reading the job descriptions.
        You can answer any question related to job description such as location, salary, benefits, ...        
    """
    SYSTEM_PROMPT = """
        Answer the question based only on the following context. If you cannot answer the question with the context, please respond with "I don't know":
    """
    USER_PROMPT = """
        Context:
        {context}

        Question:
        {question}        
    """
    chat_prompt = ChatPromptTemplate.from_messages(
        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT)
    ])
    chain = chat_prompt | openai_evalutor_model ## add pydantic parser, 
    return chain.invoke({"context": job_description, "question": job_desc_related_q})  
    
def call_model(state):
  messages = state["messages"]
  response = openai_chat_model.invoke(messages)
  return {"messages" : [response]}

def call_tool(state):
  last_message = state["messages"][-1]
  action = ToolInvocation(
      tool=last_message.additional_kwargs["function_call"]["name"],
      tool_input=json.loads(
          last_message.additional_kwargs["function_call"]["arguments"]
      )
  )
  response = tool_executor.invoke(action)
  function_message = FunctionMessage(content=str(response), name=action.tool)
  return {"messages" : [function_message]}

def should_continue(state):
  last_message = state["messages"][-1]
  if "function_call" not in last_message.additional_kwargs:
    return "end"
  return "continue"

def print_messages(messages):
  next_is_tool = False
  initial_query = True
  for message in messages["messages"]:
    if "function_call" in message.additional_kwargs:
      print()
      print(f'Tool Call - Name: {message.additional_kwargs["function_call"]["name"]} + Query: {message.additional_kwargs["function_call"]["arguments"]}')
      next_is_tool = True
      continue
    if next_is_tool:
      print(f"Tool Response: {message.content}")
      next_is_tool = False
      continue
    if initial_query:
      print(f"Initial Query: {message.content}")
      print()
      initial_query = False
      continue
    print()
    print(f"Agent Response: {message.content}")

def init():
    global openai_chat_model
    openai_chat_model = ChatOpenAI(temperature=0, top_p=0, model="gpt-3.5-turbo")
    global openai_evalutor_model
    openai_evalutor_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
    global pages
    pages = PyPDFLoader("data/resume.pdf").load_and_split()
    
    global resume
    resume = get_resume("data/resume.json")
    global summary
    summary = resume["About_Me"]["Summary"]
    global keywords
    keywords = resume["Skills"]["Technical_Skills"]
    
    jobs_list =get_jobs("data/jobs.csv")
    firstJob = jobs_list.iloc[0]
    global job_description
    job_description = firstJob.description
    
    tool_belt = [
        missing_keywords,
        inconsistencies_in_resume,
        evaluate_resume_job_description,
        modify_summary,
        adapt_resume_job_description,
        check_for_job_description_related_quires
    ]
    global tool_executor
    tool_executor = ToolExecutor(tool_belt)    
    functions = [convert_to_openai_function(t) for t in tool_belt]
    openai_chat_model = openai_chat_model.bind_functions(functions)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue" : "action",
            "end" : END
        }
    )
    workflow.add_edge("action", "agent")
    global app
    app = workflow.compile()

init()

@cl.on_chat_start
async def start_chat():    
    cl.user_session.set("app", app)

    potential_questions = [
        "List all the missing Technical talents/skills that are lacking from the resume but mentioned in the job description",
        "List all the missing non-technical talents/skills that are lacking from the resume but mentioned in the job description",
        "Enumerate every inconsistency in the resume?", 
        "Can you rate my resume for the job description?",
        "Can you suggest a better summary for my resume based on the job description?",
        "Can you adapt/modify my resume to suit the job description ?",
    ]
    await cl.Message(
        content="I guess you are here to refine your resume. Lets get to it! Here are some potential questions you can ask:",
        actions=[cl.Action(name="ask_question", value=question, label=question) for question in potential_questions]
    ).send()

@cl.on_message
async def main(message: cl.Message):
    app = cl.user_session.get("app")
    
    global job_desc_related_q
    job_desc_related_q=message.content
    
    inputs = {"messages" : [HumanMessage(content=message.content)]}
    answer = app.invoke(inputs, {"recursion_limit": 10})
    final_resp = answer["messages"][-1]
    await cl.Message(content=final_resp.content).send()

@cl.action_callback("ask_question")
async def on_action(action):
    app = cl.user_session.get("app")    
    
    inputs = {"messages" : [HumanMessage(content=action.value)]}
    answer = app.invoke(inputs, {"recursion_limit": 10})
    final_resp = answer["messages"][-1]
    await cl.Message(content=final_resp.content).send()
