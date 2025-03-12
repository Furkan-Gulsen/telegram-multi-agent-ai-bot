from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.runnables import RunnablePassthrough
import os
from langchain.agents import create_react_agent, AgentExecutor

from ..config import WAIT_TIME, MAX_MESSAGES_PER_BATCH, LLM_MODEL, LLM_TEMPERATURE
from ..database.mongodb import db
from ..utils.text import normalize_text
from ..utils.language import detect_language
from .document import document_handler

class MessageHandler:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        self.memories = {}
        self.db = db
        self.setup_conversation_chain()
        self.setup_agent_tools()
    
    def setup_conversation_chain(self):
        """Setup the conversation chain with the language model"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant capable of communicating in multiple languages and analyzing documents.

            Your tasks:
            1. Understand user messages in any language
            2. Maintain context throughout the conversation
            3. Generate helpful and logical responses
            4. Respond in the same language as the user's message
            5. ALWAYS check and use the available documents for every response
            6. When user asks about documents or their content:
               - Use the provided document context directly
               - DO NOT ask which document they're referring to
               - If the query is document-related, always use the document context
               - Reference specific parts from documents when relevant
               - If you can't find exact information, try to provide related information from documents
            
            When responding in Turkish:
            - Use ASCII characters instead of Turkish special characters
            (Replace: ğ->g, ü->u, ş->s, ı->i, ö->o, ç->c)

            Important rules:
            - Keep responses concise and clear
            - Maintain conversation context
            - Match the user's language
            - Be friendly yet professional
            - Adapt your personality to the cultural context of the language being used
            - Never ask which document to use - use all relevant document context provided
            - ALWAYS try to use document context in your responses
            - If you can't find exact information in documents, say what related information you found"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # Create chain using pipe operator
        self.chain = (
            prompt | 
            self.llm | 
            (lambda x: x.content)  # Extract content from response
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    
    def setup_agent_tools(self):
        """Setup tools for the agent"""
        self.tools = [
            Tool(
                name="Document Query",
                func=self.sync_query_documents,
                description="Search for information in user's documents"
            ),
            Tool(
                name="Language Detection",
                func=detect_language,
                description="Detect the language of user messages"
            )
        ]
        
        # Add current_user_id attribute
        self.current_user_id = None
        
        # Create agent with updated prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant that provides clear and natural responses while maintaining conversation context.

Key principles:
1. Be direct and concise
2. Use a conversational but professional tone
3. Don't mention using tools or searching documents
4. Respond naturally as if you inherently know the information
5. Keep responses focused and relevant
6. For Turkish, use ASCII characters (ğ->g, ü->u, ş->s, ı->i, ö->o, ç->c)
7. Always consider previous conversation context
8. Reference previous exchanges when relevant
9. Maintain a coherent conversation flow
10. If a question refers to previous context, use that context in your response

Available tools:
{tools}

To use a tool, use this format:
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Initialize agent with ReAct prompt
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        )
    
    def get_user_memory(self, user_id: str) -> ConversationBufferMemory:
        """Get or create memory for a user"""
        if user_id not in self.memories:
            self.memories[user_id] = ConversationBufferMemory(
                return_messages=True,
                input_key="input",
                output_key="output"
            )
        return self.memories[user_id]
    
    def get_conversation_history(self, user_id: str) -> str:
        """Get conversation history for a user"""
        # Get recent messages from MongoDB
        recent_messages = list(self.db.message_queue.find(
            {
                "user_id": user_id,
                "is_processed": True,
                "response": {"$exists": True}
            }
        ).sort("timestamp", -1).limit(5))
        
        # Format conversation history
        history = []
        for msg in reversed(recent_messages):
            history.append(f"User: {msg['message']}")
            if msg.get('response'):
                history.append(f"Assistant: {msg['response']}")
        
        return "\n".join(history) if history else ""
    
    def analyze_message_intent(self, message: str) -> str:
        """Analyze the intent of a message"""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the intent and emotion behind this message: {message}"
        )
        chain = prompt | self.llm | (lambda x: x.content)
        return chain.invoke({"message": message})
    
    async def query_documents(self, query: str, user_id: str, k: int = 5) -> Dict[str, Any]:
        """Query user's documents"""
        return await document_handler.query_documents(query, user_id, k)
    
    async def get_document_context(self, query: str, user_id: str) -> Dict[str, Any]:
        """Get relevant context from user's documents"""
        try:
            # Get user's documents from MongoDB
            user_docs = self.db.get_user_documents(user_id)
            if not user_docs:
                return {"context": "", "sources": [], "available_docs": []}

            # Format available documents info
            available_docs = []
            for doc in user_docs:
                metadata = doc.get('metadata', {})
                file_name = metadata.get('file_name', '')
                upload_time = doc.get('upload_time', datetime.utcnow()).strftime('%Y-%m-%d %H:%M')
                status = "✅" if doc.get('status') == "processed" else "❌"
                available_docs.append(f"- {status} {file_name} (Uploaded: {upload_time})")

            # Query documents with both original query and a summarization query
            doc_response = await self.query_documents(query, user_id, k=5)
            summary_response = await self.query_documents("Summarize the key points from all documents", user_id, k=3)

            # Build context with both specific and general information
            context_parts = []
            
            # Add available documents list
            context_parts.append("Your available documents:")
            context_parts.extend(available_docs)
            context_parts.append("")  # Empty line for separation

            # Add query-specific content if available
            if doc_response and doc_response.get("answer"):
                context_parts.append("Relevant content for your query:")
                context_parts.append(doc_response["answer"])
                context_parts.append("")

            # Add general summary
            if summary_response and summary_response.get("answer"):
                context_parts.append("General context from your documents:")
                context_parts.append(summary_response["answer"])

            # Combine all sources
            all_sources = []
            if doc_response and doc_response.get("sources"):
                all_sources.extend(doc_response["sources"])
            if summary_response and summary_response.get("sources"):
                # Add only unique sources from summary
                existing_ids = {s["metadata"].get("chunk_index") for s in all_sources}
                for source in summary_response["sources"]:
                    if source["metadata"].get("chunk_index") not in existing_ids:
                        all_sources.append(source)

            return {
                "context": "\n".join(context_parts),
                "sources": all_sources,
                "available_docs": available_docs,
                "stats": {
                    "total_docs": doc_response.get("total_docs", 0),
                    "docs_used": doc_response.get("docs_used", 0),
                    "total_chunks_used": len(all_sources)
                }
            }

        except Exception as e:
            print(f"Error getting document context: {str(e)}")
            return {"context": "", "sources": [], "available_docs": [], "stats": {}}
    
    async def process_messages(self, messages: List[Dict[str, Any]], user_id: str) -> str:
        """Process a batch of messages"""
        # Set current user_id for tools
        self.current_user_id = user_id
        
        # Get recent conversation history from MongoDB
        recent_history = list(self.db.message_queue.find({
            "user_id": user_id,
            "is_processed": True,
            "response": {"$exists": True}
        }).sort("timestamp", -1).limit(5))
        
        # Format conversation history for context
        conversation_context = []
        for msg in reversed(recent_history):
            conversation_context.append(f"User: {msg['message']}")
            if msg.get('response'):
                conversation_context.append(f"Assistant: {msg['response']}")
        
        # Combine messages
        combined_message = " ".join([msg["message"] for msg in messages])
        
        # Detect language
        detected_lang = detect_language(combined_message)
        
        # Normalize if Turkish
        if detected_lang == 'tr':
            combined_message = normalize_text(combined_message)
        
        # Get user memory and update with recent history
        memory = self.get_user_memory(user_id)
        for msg in conversation_context:
            if msg.startswith("User: "):
                memory.chat_memory.add_user_message(msg[6:])
            elif msg.startswith("Assistant: "):
                memory.chat_memory.add_ai_message(msg[11:])
        
        try:
            # 1. Generate semantic variations of the query
            semantic_variations = [
                combined_message,  # Original query
                f"find information about {combined_message}",
                f"what does the document say about {combined_message}",
                combined_message.replace("?", "").strip()  # Clean query
            ]
            
            # 2. Search documents with all variations
            all_responses = []
            for variation in semantic_variations:
                response = await self.query_documents(variation, user_id, k=5)
                if response and response.get("answer"):
                    all_responses.append(response)
            
            # 3. Get additional context
            doc_context = await self.get_document_context(combined_message, user_id)
            
            # 4. Prepare enhanced context
            context_parts = []
            
            # Add conversation history
            if conversation_context:
                context_parts.append("Previous Conversation:")
                context_parts.extend(conversation_context[-6:])  # Last 3 exchanges
                context_parts.append("")
            
            # Add available documents
            if doc_context['available_docs']:
                if detected_lang == 'tr':
                    context_parts.append("Mevcut Dokümanlar:")
                else:
                    context_parts.append("Available Documents:")
                context_parts.extend(doc_context['available_docs'])
                context_parts.append("")
            
            # Add relevant content from semantic searches
            if all_responses:
                for response in all_responses:
                    context_parts.append(response["answer"])
                context_parts.append("")
            
            # 5. Create enhanced message with clear instructions
            enhanced_message = f"""User Question: {combined_message}

Previous Context:
{chr(10).join(conversation_context[-6:])}

Document Context:
{chr(10).join(context_parts)}

Instructions:
1. Consider the previous conversation context when responding
2. Provide a direct and natural response
3. Don't mention that you're using documents or sources
4. Keep the response concise and focused
5. Use a conversational but professional tone
6. Respond in {detected_lang} language
7. If information isn't available, say so briefly
8. For Turkish responses, use ASCII characters (ğ->g, ü->u, ş->s, ı->i, ö->o, ç->c)
"""
            
            # 6. Use agent with enhanced context
            agent_response = await self.agent.ainvoke({
                "input": enhanced_message,
                "chat_history": memory.chat_memory.messages
            })
            response = agent_response.get("output", "")
            
            # 7. Update conversation history with document usage information
            self.db.message_queue.update_one(
                {"_id": messages[-1]["_id"]},
                {
                    "$set": {
                        "conversation_history": {
                            "user_message": combined_message,
                            "assistant_response": response,
                            "language": detected_lang,
                            "document_context_used": True,
                            "previous_context_used": bool(conversation_context),
                            "documents_referenced": [
                                source["metadata"]["file_name"]
                                for source in doc_context.get("sources", [])
                            ],
                            "semantic_variations_used": len(all_responses),
                            "total_chunks_used": len(doc_context.get("sources", [])),
                            "timestamp": datetime.utcnow()
                        }
                    }
                }
            )
            
            return response
            
        except Exception as e:
            print(f"Error in process_messages: {str(e)}")
            # Fall back to direct document content if available
            if all_responses and all_responses[0].get("answer"):
                return all_responses[0]["answer"]
            
            if detected_lang == 'tr':
                return "Üzgünüm, dokümanlarınızda ilgili bir bilgi bulamadım veya bir hata oluştu."
            else:
                return "Sorry, I couldn't find relevant information in your documents or an error occurred."
    
    async def process_message_queue(self, user_id: str):
        """Process messages in the queue for a user"""
        try:
            # Wait for additional messages
            await asyncio.sleep(WAIT_TIME)
            
            # Get cutoff time
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(minutes=5)
            
            # Get pending messages
            messages = self.db.get_pending_messages(user_id, cutoff_time, MAX_MESSAGES_PER_BATCH)
            
            if not messages:
                return
            
            # Generate batch ID
            batch_id = str(datetime.utcnow())
            message_ids = [msg["_id"] for msg in messages]
            
            # Mark messages as being processed
            self.db.mark_messages_as_processed(message_ids, batch_id)
            
            # Process messages with RAG
            response = await self.process_messages(messages, user_id)
            
            # Update messages with response
            self.db.update_message_response(batch_id, response)
            
            return response
            
        except Exception as e:
            print(f"Error processing messages for user {user_id}: {str(e)}")
            if 'message_ids' in locals():
                self.db.message_queue.update_many(
                    {"_id": {"$in": message_ids}},
                    {
                        "$set": {
                            "processing_error": str(e),
                            "error_timestamp": datetime.utcnow()
                        }
                    }
                )
            raise
    
    def sync_query_documents(self, query: str) -> Dict[str, Any]:
        """Synchronous version of query_documents for the agent tool"""
        try:
            # Create an event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function in the event loop
            result = loop.run_until_complete(
                document_handler.query_documents(query, self.current_user_id)
            )
            
            # Format the result for better readability
            if isinstance(result, dict):
                formatted_result = {
                    "answer": result.get("answer", "No relevant information found."),
                    "sources": [
                        f"From {s['metadata'].get('file_name', 'Unknown')}: {s['content'][:200]}..."
                        for s in result.get("sources", [])
                    ]
                }
                return formatted_result
            return {"answer": "No results found", "sources": []}
            
        except Exception as e:
            print(f"Error in sync_query_documents: {str(e)}")
            return {"answer": "Error querying documents", "sources": []}

# Create a singleton instance
message_handler = MessageHandler()
