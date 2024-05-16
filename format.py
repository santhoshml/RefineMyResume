from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional,Union,Any
from pydantic import BaseModel, Field

class OutputFormat(BaseModel):
    Contact_Information: Optional[Any] = None
    About_Me: Optional[Any] = None
    Work_Experience: Optional[Any] = None
    Education: Optional[Any] = None
    Skills: Optional[Any] = None
    Certificates: Optional[Any] = None
    Projects: Optional[Any] = None
    Achievements: Optional[Any] = None
    Interests:Optional[Any]=None
    Volunteer: Optional[Any] = None

class SkillsFormat(BaseModel):    
    Skills: Optional[Any] = None