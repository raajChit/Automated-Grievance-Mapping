
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

def initialize_llm(model_provider, model_name):
    if model_provider == "groq":
        llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="ENTER API KEY HERE"
        )
    elif model_provider == "anthropic":
        llm = ChatAnthropic(
        model=model_name,
        api_key="ENTER API KEY HERE"
        )
    elif model_provider == "together_ai":
        llm  = ChatTogether(
        model=model_name,
        together_api_key="ENTER API KEY HERE"
        )
    elif model_provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
            api_key="ENTER API KEY HERE"
        )
    else:
        return "Wrong llm name and model"
    return llm