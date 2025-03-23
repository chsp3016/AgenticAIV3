import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        st.info(f"Processing request for usecase: '{usecase}'")
        print(f"Debug: Selected usecase is '{usecase}'")
        
        # Display user message first for all usecases
        with st.chat_message("user"):
            st.write(user_message)
        
        if usecase == "Basic Chatbot":
            # Create a container for the assistant's message
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use a properly formatted human message
                human_msg = HumanMessage(content=user_message)
                
                try:
                    # Stream the response
                    for chunk in graph.stream({"messages": [human_msg]}):
                        print(f"Debug: Received chunk: {chunk}")
                        if "messages" in chunk:
                            last_msg = chunk["messages"][-1] if isinstance(chunk["messages"], list) else chunk["messages"]
                            if isinstance(last_msg, AIMessage) and hasattr(last_msg, "content"):
                                content = last_msg.content or ""
                                full_response += content
                                message_placeholder.write(full_response)
                    
                    # If no response was generated
                    if not full_response:
                        message_placeholder.write("No response generated. Please check your configuration.")
                except Exception as e:
                    message_placeholder.write(f"Error processing request: {str(e)}")
                    print(f"Error in Basic Chatbot: {str(e)}")

        elif usecase == "Chatbot with Tool":
            try:
                # Prepare state and invoke the graph
                initial_state = {"messages": [HumanMessage(content=user_message)]}
                print(f"Debug: Invoking graph with initial state: {initial_state}")
                
                res = graph.invoke(initial_state)
                print(f"Debug: Full response from graph: {res}")
                
                # Check if we have a valid response
                if not res.get('messages', []):
                    with st.chat_message("assistant"):
                        st.error("No messages returned from the model. Please check your configuration.")
                    return
                
                # Process and display all messages in the response
                for i, message in enumerate(res['messages']):
                    print(f"Debug: Message {i} type: {type(message)}")
                    
                    if isinstance(message, HumanMessage):
                        # Skip redisplaying user message
                        print(f"Debug: Skipping Human message {i}")
                        pass
                    elif isinstance(message, ToolMessage):
                        with st.chat_message("tool"):
                            st.write("Tool Call Start")
                            st.write(message.content)
                            st.write("Tool Call End")
                    elif isinstance(message, AIMessage):
                        with st.chat_message("assistant"):
                            if message.content:
                                st.write(message.content)
                            else:
                                st.write("(Empty message received)")
                    else:
                        with st.chat_message("system"):
                            st.write(f"Unknown message type: {type(message)}")
                            if hasattr(message, 'content'):
                                st.write(message.content)
                            else:
                                st.write(str(message))
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Error processing request: {str(e)}")
                print(f"Error in Chatbot with Tool: {str(e)}")
        
        elif usecase == "Blog Creation Tool":
            try:
                # Prepare state and invoke the graph
                initial_state = {"messages": [HumanMessage(content=user_message)]}
                print(f"Debug: Invoking graph with initial state: {initial_state}")
                
                res = graph.invoke(initial_state)
                print(f"Debug: Full response from graph: {res}")
                
                # Check if we have a valid response
                if not res.get('messages', []):
                    with st.chat_message("assistant"):
                        st.error("No messages returned from the model. Please check your configuration.")
                    return
                
                # Process and display all messages in the response
                for i, message in enumerate(res['messages']):
                    print(f"Debug: Message {i} type: {type(message)}")
                    
                    if isinstance(message, HumanMessage):
                        # Skip redisplaying user message
                        print(f"Debug: Skipping Human message {i}")
                    elif isinstance(message, AIMessage):
                        with st.chat_message("assistant"):
                            if message.content:
                                st.markdown(message.content)
                            else:
                                st.write("(Empty message received)")
                    else:
                        with st.chat_message("system"):
                            st.write(f"Unknown message type: {type(message)}")
                            if hasattr(message, 'content'):
                                st.write(message.content)
                            else:
                                st.write(str(message))
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Error processing request: {str(e)}")
                print(f"Error in Blog Creation Tool: {str(e)}")
        
        elif usecase.strip().lower() == "code review tool".lower():
            try:
                # Prepare state and invoke the graph
                initial_state = {
                    "messages": [HumanMessage(content=user_message)],
                    "topic": user_message,  # Set the requirement/topic as string
                    "created_code": "",
                    "review_peer": "",
                    "review_manager": ""
                }
                
                print(f"Debug: Invoking Code Review graph with initial state: {initial_state}")
                res = graph.invoke(initial_state)
                print(f"Debug: Full response from graph: {res}")
                print(f"Debug: Response keys: {res.keys() if isinstance(res, dict) else 'Not a dict'}")
                
                # Check if we have a valid response
                if not res.get('messages', []):
                    with st.chat_message("assistant"):
                        st.error("No messages returned from the model. Please check your configuration.")
                    return
                
                # Debug info for messages
                print(f"Debug: Number of messages: {len(res.get('messages', []))}")
                for i, message in enumerate(res.get('messages', [])):
                    print(f"Debug: Message {i} type: {type(message)}")
                    print(f"Debug: Message {i} attributes: {dir(message) if hasattr(message, '__dict__') else 'No attributes'}")
                    if hasattr(message, 'content'):
                        print(f"Debug: Message {i} content type: {type(message.content)}")
                        print(f"Debug: Message {i} content preview: {message.content[:100] if message.content else 'Empty'}")
                
                # Process all the messages in the response
                message_count = {"code_generator": 0, "peer_reviewer": 0, "manager_reviewer": 0}
                
                for i, message in enumerate(res.get('messages', [])):
                    print(f"Debug: Processing message {i}, type: {type(message)}")
                    
                    if isinstance(message, HumanMessage):
                        # Skip redisplaying user messages
                        print(f"Debug: Skipping Human message {i}")
                    
                    elif isinstance(message, AIMessage) and hasattr(message, 'content') and message.content:
                        # For the first AI message, assume it's code generation
                        if message_count["code_generator"] == 0:
                            with st.chat_message("code_generator"):
                                st.markdown("**Generated Code:**")
                                st.code(message.content, language="python")
                                res["created_code"] = message.content
                                message_count["code_generator"] += 1
                        
                        # For the second AI message, assume it's peer review
                        elif message_count["peer_reviewer"] == 0:
                            with st.chat_message("peer_reviewer"):
                                st.markdown("**Peer Review:**")
                                st.markdown(message.content)
                                res["review_peer"] = message.content
                                message_count["peer_reviewer"] += 1
                        
                        # For the third AI message, assume it's manager review
                        elif message_count["manager_reviewer"] == 0:
                            with st.chat_message("manager_reviewer"):
                                st.markdown("**Manager Review:**")
                                st.markdown(message.content)
                                res["review_manager"] = message.content
                                message_count["manager_reviewer"] += 1
                        
                        # For any other AI message
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(message.content)
                    
                    elif isinstance(message, AIMessage) and (not hasattr(message, 'content') or not message.content):
                        with st.chat_message("assistant"):
                            st.write("(Empty message received)")
                    
                    else:
                        # Handle any other type of message
                        with st.chat_message("system"):
                            st.write(f"Message of type: {type(message)}")
                            if hasattr(message, 'content'):
                                st.write(message.content)
                            else:
                                st.write(str(message))
                
                # Display summary if no content was displayed
                if sum(message_count.values()) == 0:
                    with st.chat_message("assistant"):
                        st.error("No valid AI messages were found in the response.")
                        st.write("Please check your LangChain graph configuration.")
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Error processing request: {str(e)}")
                import traceback
                print(f"Error in Code Review Tool: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
        
        else:
            with st.chat_message("assistant"):
                st.error(f"Unknown usecase: {usecase}")