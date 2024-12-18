from dotenv import load_dotenv
load_dotenv("../../.env.local")

from datetime import datetime


from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from schemas import AnswerQuestion

llm = ChatOpenAI(model_name="gpt-4o-mini")
# llm = ChatAnthropic(model_name="claude-3-5-sonnet-latest")

# llm = ChatOllama(model="mistral-nemo:latest")
# llm_model = "llama3.1:8b"

parser_json = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_promt_template = ChatPromptTemplate.from_messages(
[
    SystemMessage(content=
     """You are an expert researcher.
     Current time: {time}
     1. {first_instruction}
     2. Reflect and critique your answer. Be severe to maximize improvement.
     3. Recommend search queeries to research information and imporove your ansewr"""),
     SystemMessage(content="Answer the user's question below using the required format"),
     MessagesPlaceholder(variable_name="messages"),
]
).partial(time=lambda: datetime.now().isoformat())

first_responder_promt_template = actor_promt_template.partial(
    first_instruction="Provide a detailed ~250 word answer")

first_responder = first_responder_promt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

 

def format_response(response):
    """Format the response for better readability."""
    print("\n=== Answer ===\n")
    print(response.answer)
    
    print("\n=== Reflection ===\n")
    print("Missing Information:")
    for item in response.reflection.missing_information:
        print(f"- {item}")
    
    print("\nSuperfluous Information:")
    for item in response.reflection.superflous_information:
        print(f"- {item}")
        
    print("\nImprovement Suggestions:")
    for item in response.reflection.improvement_suggestions:
        print(f"- {item}")
    
    print("\n=== Search Queries ===\n")
    for query in response.search_queries:
        print(f"- {query}")


if __name__ == "__main__":
    print("Starting actor chain...")
    human_message = HumanMessage(content="how to make viral content")
    # chain = first_responder_promt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic #this is the old way
    chain = first_responder_promt_template | llm.with_structured_output(AnswerQuestion) #this is the new way    
    res = chain.invoke({"messages": [human_message]})
    # print(res)
    format_response(res)

