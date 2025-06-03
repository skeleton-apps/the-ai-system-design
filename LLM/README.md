# LLM System Design 
## Functional Requirements
- User Should be able to input natural language text as input and get relevant response from a Large Language Model (LLM)
- The system should maintain context across multi-turn conversations for each user
- The system should be able to retrieve relevant documents or facts from external sources (RAG) to enhance responses
- The System should support different assistant roles 
- The system should filter or block unsafe, harmful or inappropriate content
- Users should be able to provide a feedback on responses to improve system performance
## Non-Functional Requirements
- The system should respond to user queries within 1 second for standard prompts (Low Latency)
- The system should handle up to 10,000 concurrent users without performance degradation 
- The system should ensure 99.9 uptime (High Availability)
- The system should be resilient to failure with automatic recovery and recovery
## Defining the Core Entities 

- User 
	-	`user_id` (UUID)    
	-	`username`    
	-   `email`    
	-   `created_at`    
	-   `last_login_at`    
	-   `preferences` (JSON)   
	
- Conversation

	-   `conversation_id` (UUID)    
	-   `user_id` (FK → User)    
	-   `created_at`    
	-   `last_updated_at`    
	-   `assistant_role_id` (FK → AssistantRole)    
	-   `status` (active, archived)
    
- Message
	-   `message_id` (UUID)    
	-   `conversation_id` (FK → Conversation)    
	-   `sender` (enum: user, assistant)    
	-   `content` (text)    
	-   `timestamp`    
	-   `response_latency_ms`    
	-   `retrieved_docs` (optional JSON list for RAG traceability)    

- AssistantRole
	-   `role_id` (UUID)    
	-   `name` (e.g., "Travel Assistant", "Medical Advisor")    
	-   `description`    
	-   `system_prompt` (LLM configuration per role)    

- Feedback
	-   `feedback_id` (UUID)    
	-   `message_id` (FK → Message)    
	-   `user_id` (FK → User)    
	-   `rating` (enum: thumbs_up, thumbs_down, neutral)    
	-   `comment` (optional)   
	-  
- Document (RAG Source)
	-   `doc_id` (UUID)    
	-   `source_name`    
	-   `content` (text or embedding reference)    
	-   `metadata` (JSON – e.g., date, tags, type)    
	-   `indexed_at`
