from datetime import datetime
import streamlit as st
import validators
from playwright.async_api import async_playwright
from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from tarsier import Tarsier, GoogleVisionOCRService
import json
import os
import asyncio
from langchain.agents import tool
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv("../.env.local")

class NavigationElement(BaseModel):
    element_id: int = Field(description="Unique identifier for the element on the page")
    text: str = Field(description="Visible text of the navigation element")
    element_type: str = Field(description="Type of element (link, button, menu)")
    xpath: str = Field(description="XPath location of the element")
    target_url: Optional[str] = Field(None, description="URL that this element navigates to, if applicable")

class InteractiveElement(BaseModel):
    element_id: int = Field(description="Unique identifier for the element on the page")
    element_type: str = Field(description="Type of interactive element (form, input, dropdown)")
    label: Optional[str] = Field(None, description="Label or placeholder text for the element")
    xpath: str = Field(description="XPath location of the element")

class ContentStructure(BaseModel):
    main_heading: Optional[str] = Field(None, description="Main heading of the page")
    sub_headings: List[str] = Field(default_factory=list, description="List of subheadings on the page")
    main_content: str = Field(description="Main textual content of the page")

class PageNode(BaseModel):
    url: str = Field(description="URL of the page")
    title: str = Field(description="Title of the page")
    timestamp: datetime = Field(description="Time when the page was analyzed")
    navigation_elements: List[NavigationElement] = Field(
        default_factory=list,
        description="List of elements that can be used for navigation"
    )
    content_structure: ContentStructure = Field(
        description="Structure and content of the page"
    )
    interactive_elements: List[InteractiveElement] = Field(
        default_factory=list,
        description="List of interactive elements on the page"
    )

def setup_tarsier():
    # Setup Tarsier with Google Vision OCR
    with open("../.tarsier.json", "r") as f:   
        google_cloud_credentials = json.load(f)
    ocr_service = GoogleVisionOCRService(google_cloud_credentials)
    return Tarsier(ocr_service)

def setup_agent(tarsier, page):
    tag_to_xpath = {}
    
    # Define tools/actions
    @tool (response_format="content_and_artifact")
    async def read_page():
        """Use to read the current state of the page"""
        return await read_page_impl()
    
    async def read_page_impl() -> str:
        page_text, inner_tag_to_xpath = await tarsier.page_to_text(page)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        await page.screenshot(path=f'../screenshots/${now}.png')
        tag_to_xpath.clear()
        tag_to_xpath.update(inner_tag_to_xpath)
        with open(f'../screenshots/${now}.png', 'rb') as image_file:
            image_data = image_file.read()  
        return page_text, image_data
    
    @tool(response_format="content_and_artifact")
    async def click(element_id: int) -> str:
        """
        Click on an element based on element_id and return the new page state
        """
        x_path = tag_to_xpath[element_id]['xpath']
        print(x_path)
        element = page.locator(x_path)
        await element.scroll_into_view_if_needed()
        await page.wait_for_timeout(1000)
        await element.click()
        await page.wait_for_timeout(2000)
        return await read_page_impl()
    
    @tool (response_format="content_and_artifact")
    async def type_text(element_id: int, text: str) -> str:
        """
        Input text into a textbox based on element_id and return the new page state
        """
        x_path = tag_to_xpath[element_id]['xpath']
        print(x_path)
        try:
            await page.locator(x_path).clear()
        except Exception as e:
            print(e)
        await page.locator(x_path).press_sequentially(text)
        return await read_page_impl()


    @tool (response_format="content_and_artifact")
    async def press_key(key: str) -> str:
        """
        Press a key on the keyboard and return the new page state
        """
        await page.keyboard.press(key)
        await page.wait_for_timeout(2000)
        return await read_page_impl()


    template = """
    You are a web interaction agent. Use the read page tool to understand where you currently are. 
    You will be passed in OCR text of a web page and its screenshot where element ids are to the left of elements. 

    You have access to the following tools:
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    These were previous tasks you completed:

    Begin!

    Question: {input}
    {agent_scratchpad}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Setup LLM and agent
    llm = ChatAnthropic(model_name="claude-3-5-sonnet-latest", temperature=0)
    agent = initialize_agent(
        [read_page, click, type_text, press_key],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

async def process_url(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        try:
            # Navigate to URL
            await page.goto(url)
            st.write("Successfully navigated to URL")
            
            # Setup Tarsier and agent
            tarsier = setup_tarsier()
            agent = setup_agent(tarsier, page)
            
            # Analyze page content
            st.write("Analyzing page content...")
            analysis = await agent.arun("Read and analyze the current page content")
            st.write("Analysis:", analysis)
            
        except Exception as e:
            st.error(f"Error accessing URL: {str(e)}")
        finally:
            await browser.close()

def main():
    st.title("URL Discovery Tool")
    st.markdown("Enter a URL below to analyze it")
    
    url = st.text_input("Enter URL:", placeholder="https://example.com")
    
    if url:
        if validators.url(url):
            st.success("Valid URL entered!")
            # Run async code in sync context
            asyncio.run(process_url(url))
        else:
            st.error("Please enter a valid URL")

if __name__ == "__main__":
    main()
