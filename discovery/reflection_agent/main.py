from dotenv import load_dotenv
load_dotenv("../../.env.local")


import os
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generation_chain, reflection_chain



REFLECT = "reflect"
GENERATE = "generate"

def generation_node(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return generation_chain.invoke({"messages": messages})


def reflection_node(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(messages: Sequence[BaseMessage]) -> str:
    if len(messages) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# print(graph.get_graph().print_ascii())

    


if __name__ == "__main__":
    inputs = HumanMessage(content="Generate a tweet about the future of AI")
    res = graph.invoke([inputs])
    print(res[0].content)
    print(res[1].content)
    print(res[-1].content)
