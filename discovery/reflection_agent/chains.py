from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and suggestions for improvement."\
            "Always provide a detailed recommendations, including request for length, virality, style, tone, etc."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer generating a tweet."
            "You will be asked to generate a tweet and optionally some critique of the tweet. Generate a revised version of the tweet that addresses the critique, unless directed otherwise."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

