import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import sys
import shutil
from datetime import datetime
import pytz


# llm_model = "mistral-nemo:latest"
llm_model = "llama3.1:8b"


@tool
def get_disk_usage():
    """Get disk usage information for the root directory.

    Args:
        None

    Returns:
        dict: A dictionary containing disk usage information with the following keys:
            - total_gb (float): Total disk space in gigabytes
            - used_gb (float): Used disk space in gigabytes
            - free_gb (float): Free disk space in gigabytes
    """
    print("Getting disk usage...")
    total, used, free = shutil.disk_usage("/")
    # Convert bytes to GB
    total_gb = total / (2**30)  # 1 GB = 2^30 bytes
    used_gb = used / (2**30)
    free_gb = free / (2**30)

    # Return as JSON/dict
    disk_info = {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "free_gb": round(free_gb, 2),
    }
    return disk_info


@tool
def get_time_in_timezone(timezone: str):
    """Get the current time in a specific timezone.

    Args:
        timezone (str): The timezone to get the time in (e.g., 'US/Pacific', 'Europe/London').

    Returns:
        str: The current time in the specified timezone.
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except pytz.exceptions.UnknownTimeZoneError:
        return (
            f"Error: Unknown timezone '{timezone}'. Please use a valid timezone name."
        )


# Initialize session state for message history if it doesn't exist
if "message_history" not in st.session_state:
    st.session_state.message_history = []

messages = []

# Create two columns - left for history, right for chat interface
left_col, right_col = st.columns([1, 2])

# Display message history in the left column
with left_col:
    st.markdown("### Conversation History")
    history_container = st.container(height=600, border=True)
    with history_container:
        for msg in st.session_state.message_history:
            if isinstance(msg, HumanMessage):
                st.markdown("**You:** " + msg.content)
            elif isinstance(msg, AIMessage):
                st.markdown("**Assistant:** " + msg.content)
            elif isinstance(msg, ToolMessage):
                st.markdown("**Tool Response:** " + str(msg.content))
            st.markdown("---")

# Chat interface in the right column
with right_col:
    tools_list = {
        "get_disk_usage": get_disk_usage,
        "get_time_in_timezone": get_time_in_timezone,
    }

    # Create a form for input
    with st.form(key="chat_form"):
        prompt = st.text_input("Enter your prompt (type 'exit' to quit)")
        submit_button = st.form_submit_button("Send")

    if submit_button and prompt:
        if prompt.lower() == "exit":
            st.write("Goodbye!")
            st.stop()

        llm = ChatOllama(model=llm_model)
        llm_with_tools = llm.bind_tools(list(tools_list.values()))
        human_message = HumanMessage(content=prompt)
        messages.append(human_message)
        st.session_state.message_history.append(human_message)

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            ai_message = AIMessage(content=response.content)
            messages.append(ai_message)
            st.session_state.message_history.append(ai_message)
            with st.container(height=500, border=True):
                st.write(response.content)

        else:
            for tool_call in response.tool_calls:
                if tool_call["name"].lower() not in tools_list:
                    error_message = AIMessage(
                        content=f"Error: Tool '{tool_call['name']}' not found."
                    )
                    messages.append(error_message)
                    st.session_state.message_history.append(error_message)
                    with st.container(height=500, border=True):
                        st.write(f"Error: Tool '{tool_call['name']}' not found.")
                    continue

                selected_tool = tools_list.get(tool_call["name"].lower())
                print(selected_tool)
                tool_response = selected_tool.invoke(tool_call["args"])
                print("tool_response: ", tool_response)
                tool_message = ToolMessage(tool_response, tool_call_id=tool_call["id"])
                messages.append(tool_message)
                st.session_state.message_history.append(tool_message)

            response = llm_with_tools.stream(messages)
            with st.container(height=500, border=True):
                st.write(response)
