from Descriptive_Answer_processor import *
from RAG_Processor import *

def Get_Predictive_Answer(question):
    return "predictive"

def Get_Prescriptive_Answer(question):
    return "prescriptive"

def Get_Descriptive_Answer(question):
    Get_RAG_Answer(question)
    return "descriptive answer"

def Get_Answer(question, question_type):
    if "predictive" in question_type:
        return Get_Predictive_Answer(question)
    elif "prescriptive" in question_type:
        return Get_Prescriptive_Answer(question)
    else:
        return Get_Descriptive_Answer(question)