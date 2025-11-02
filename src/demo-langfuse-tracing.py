"""
Simple Traced Chatbot with Langfuse
====================================

This chatbot does three things:
1. Takes a question from the user
2. Asks OpenAI's GPT-3.5 for an answer
3. Logs everything to Langfuse so you can see what happened

Think of it like a restaurant where:
- Your app is the customer ordering food
- OpenAI is the kitchen making the food
- Langfuse is the manager taking notes about everything

Let's build it step by step!
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from langfuse import Langfuse
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid

# ============================================================================
# STEP 0: Load Environment Variables
# ============================================================================
# This automatically loads your .env file so you don't have to export manually
# It's like having a helper read your recipe card before you start cooking

load_dotenv()  # Loads variables from .env file into environment
print("✓ Environment variables loaded from .env file")

# ============================================================================
# STEP 1: Initialize Our Tools
# ============================================================================
# Think of this as gathering all your ingredients before cooking

app = FastAPI(title="Traced Chatbot")

# Connect to OpenAI (the kitchen that makes answers)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✓ Connected to OpenAI")

# Connect to Langfuse (the manager taking notes)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)
print("✓ Connected to Langfuse")

# ============================================================================
# STEP 2: Define What Data Looks Like
# ============================================================================
# These are like order forms - they specify what information we need

class ChatRequest(BaseModel):
    """What the user sends us"""
    message: str           # The question they're asking
    user_id: str = "anonymous"  # Who's asking (optional)

class ChatResponse(BaseModel):
    """What we send back to the user"""
    response: str          # The answer from GPT
    trace_id: str          # A unique ID to find this conversation later
    trace_url: str         # A link to see details in Langfuse
    tokens_used: int       # How many tokens (words) GPT used
    model: str             # Which AI model answered

# ============================================================================
# STEP 3: The Main Chat Function
# ============================================================================
# This is where the magic happens!

@app.post("/chat")
def chat(request: ChatRequest):
    """
    The main function that handles a chat request.
    
    Think of this like a restaurant waiter:
    1. Takes the order (receives the question)
    2. Gives order to kitchen (sends to OpenAI)
    3. Delivers food (returns the answer)
    4. Tells manager what happened (logs to Langfuse)
    """
    
    # Generate a unique ID for this conversation
    # Like giving each restaurant order a ticket number
    trace_id = str(uuid.uuid4())
    
    # Start logging this conversation to Langfuse
    # The "trace" is like the order ticket - it tracks the whole conversation
    trace = langfuse.trace(
        id=trace_id,
        name="chat_conversation",
        user_id=request.user_id,
        input={"message": request.message},
        metadata={
            "timestamp": datetime.now().isoformat(),
            "message_length": len(request.message)
        },
        tags=["chatbot", "gpt-3.5"]
    )
    
    print(f"📝 Started tracking conversation: {trace_id}")
    
    # Call OpenAI to get the answer
    # This is like sending the order to the kitchen
    llm_response = call_openai(
        message=request.message,
        trace=trace
    )
    
    # Process the response (in this simple version, we just pass it through)
    # In a real app, you might check for bad words, format it nicely, etc.
    final_response = process_response(
        llm_response=llm_response,
        trace=trace
    )
    
    # Update the trace with the final result
    # Like the manager noting "Order completed successfully!"
    trace.update(
        output={
            "response": final_response["text"],
            "tokens": final_response["tokens"]
        }
    )
    
    # Send the data to Langfuse immediately (don't wait)
    langfuse.flush()
    
    print(f"✅ Conversation completed: {trace_id}")
    
    # Build the URL where the user can see details
    trace_url = f"{os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}/trace/{trace_id}"
    
    # Return the answer to the user
    return ChatResponse(
        response=final_response["text"],
        trace_id=trace_id,
        trace_url=trace_url,
        tokens_used=final_response["tokens"],
        model=final_response["model"]
    )

# ============================================================================
# STEP 4: Call OpenAI (The Kitchen)
# ============================================================================

def call_openai(message: str, trace):
    """
    Sends the question to OpenAI and gets an answer.
    
    This is like the kitchen receiving an order and cooking it.
    We use a "generation" to track this specific LLM call.
    
    A generation is a special type of span designed for LLM calls.
    It automatically tracks tokens, costs, and model info!
    """
    
    print(f"  🤖 Asking GPT-3.5: {message[:50]}...")
    
    # Actually call OpenAI's API
    # This is where the magic happens - GPT generates the answer
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that explains things simply."
            },
            {
                "role": "user", 
                "content": message
            }
        ],
        temperature=0.7,
        max_tokens=500  # Maximum length of the answer
    )
    
    # Extract what we need from OpenAI's response
    result = {
        "text": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "model": response.model
    }
    
    # Log this as a GENERATION (not a regular span)
    # This tells Langfuse "this is an LLM call, track tokens and cost!"
    trace.generation(
        name="openai_api_call",
        model=response.model,  # e.g., "gpt-3.5-turbo-0125"
        input=message,
        output=result["text"],
        usage={
            "promptTokens": response.usage.prompt_tokens,      # Tokens in your question
            "completionTokens": response.usage.completion_tokens,  # Tokens in GPT's answer
            "totalTokens": response.usage.total_tokens        # Total (input + output)
        },
        metadata={
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    print(f"  ✓ Got answer ({result['tokens']} tokens)")
    
    return result

# ============================================================================
# STEP 5: Process the Response
# ============================================================================

def process_response(llm_response: dict, trace):
    """
    Processes OpenAI's response before sending it to the user.
    
    In this simple version, we just pass it through unchanged.
    In a real app, you might:
    - Check for inappropriate content
    - Add citations or sources
    - Format it nicely
    - Translate to another language
    
    We still create a span to track this step, even though it's simple.
    """
    
    # Create another span for this processing step
    span = trace.span(
        name="response_processing",
        input=llm_response,
        metadata={"step": "validation"}
    )
    
    print(f"  🔍 Processing response...")
    
    # In this simple example, we don't modify anything
    # Just pass it through as-is
    processed = {
        "text": llm_response["text"],
        "tokens": llm_response["tokens"],
        "model": llm_response["model"]
    }
    
    # Mark this step as complete
    span.end(
        output=processed,
        metadata={"status": "passed"}
    )
    
    print(f"  ✓ Processing complete")
    
    return processed

# ============================================================================
# STEP 6: Health Check and Info Endpoints
# ============================================================================
# These are utility endpoints to check if everything is working

@app.get("/health")
def health():
    """
    Simple endpoint to check if the service is alive.
    Like asking "Is anyone home?"
    """
    return {
        "status": "healthy",
        "message": "Chatbot is running!"
    }

@app.get("/")
def root():
    """
    Root endpoint that shows usage instructions.
    Like the front door with a sign explaining what's inside.
    """
    return {
        "message": "Welcome to the Traced Chatbot!",
        "usage": {
            "endpoint": "POST /chat",
            "example": {
                "message": "What is observability?",
                "user_id": "demo_user"
            }
        },
        "view_traces": f"{os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}/traces"
    }

# ============================================================================
# STEP 7: Start the Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 STARTING TRACED CHATBOT")
    print("="*60)
    
    # Check if we have all the required API keys
    required_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY")
    }
    
    missing = [key for key, value in required_keys.items() if not value]
    
    if missing:
        print(f"\n❌ Missing required environment variables:")
        for key in missing:
            print(f"   - {key}")
        print("\nSet them with:")
        print("  export OPENAI_API_KEY='sk-your-key'")
        print("  export LANGFUSE_PUBLIC_KEY='pk-lf-your-key'")
        print("  export LANGFUSE_SECRET_KEY='sk-lf-your-key'")
        exit(1)
    
    print("\n✅ All environment variables set!")
    print(f"\n📊 View traces at: {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}/traces")
    print(f"\n🌐 Starting server on http://localhost:8000")
    print("\n📝 Test with:")
    print('   curl -X POST http://localhost:8000/chat \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message":"What is AI?","user_id":"test"}\'')
    print("\n" + "="*60 + "\n")
    
    # Start the web server
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ============================================================================
# HOW IT ALL WORKS TOGETHER (The Big Picture)
# ============================================================================
"""
When a user sends a message, here's what happens:

1. REQUEST ARRIVES
   User → POST /chat with their question
   
2. CREATE TRACE
   We generate a unique ID and start logging to Langfuse
   Think: "Start a new order ticket #12345"
   
3. CALL OPENAI (Span #1)
   - Log: "Started asking GPT"
   - Call OpenAI API with the question
   - Get back the answer
   - Log: "Got answer, used X tokens"
   
4. PROCESS RESPONSE (Span #2)
   - Log: "Started processing"
   - Check/clean the response (in this simple version, do nothing)
   - Log: "Processing complete"
   
5. FINISH TRACE
   - Update the main trace with the final result
   - Send everything to Langfuse
   
6. RETURN TO USER
   - Send back the answer
   - Include a link to see the trace details

The result? You can now see:
- How long each step took
- How many tokens were used
- What the input and output were
- If any errors occurred
- Who asked the question and when

All without cluttering your code with tons of logging statements!
"""

# ============================================================================
# KEY CONCEPTS EXPLAINED
# ============================================================================
"""
🎯 TRACE
--------
A trace is the whole conversation from start to finish.
Like a movie from opening credits to the end.

Example: User asks "What is AI?" → Get answer → Return it
That entire flow is ONE trace.


🎯 SPAN
-------
A span is one step within a trace.
Like a scene in the movie.

There are TWO types of spans:

1. REGULAR SPAN (for general operations):
   - Database queries
   - API calls (non-LLM)
   - Processing steps
   
   Example:
   trace.span(
       name="database_query",
       input={"query": "SELECT ..."}
   )

2. GENERATION (special span for LLM calls):
   - OpenAI API calls
   - Anthropic API calls
   - Any LLM inference
   
   Example:
   trace.generation(
       name="openai_call",
       model="gpt-3.5-turbo",
       usage={"promptTokens": 20, "completionTokens": 85}
   )

**Why use generation() for LLM calls?**
- ✅ Automatically tracks token usage
- ✅ Calculates costs based on model pricing
- ✅ Shows up specially in Langfuse UI
- ✅ Includes model information

Each span/generation has:
- Start and end time
- Input and output
- Metadata (extra info)
- Duration (automatically calculated)


🎯 METADATA
-----------
Extra information about what happened.
Like writing notes in the margin of a recipe.

Example:
- Which model we used (gpt-3.5-turbo)
- What temperature setting (0.7)
- How long the message was


🎯 TOKENS
---------
How OpenAI measures usage. Roughly:
- 1 token ≈ 4 characters
- "Hello world" ≈ 2 tokens

OpenAI charges by tokens:
- Input tokens: Your question
- Output tokens: GPT's answer
- Total = Input + Output


🎯 FLUSH
--------
Telling Langfuse "send the data NOW, don't wait!"

By default, Langfuse batches data and sends it every few seconds.
Calling flush() says "I want to see it immediately!"


🎯 THE RESTAURANT ANALOGY
-------------------------
Your App    = Customer (places order)
FastAPI     = Waiter (takes order, serves food)
OpenAI      = Kitchen (cooks the food)
Langfuse    = Manager (writes everything down)

Trace = The complete order (from walking in to leaving)
Span  = Each step (ordering, cooking, serving)

The manager (Langfuse) tracks:
- What time the customer arrived
- What they ordered
- How long it took to cook
- Whether they were satisfied
- How much it cost

This way, the restaurant can:
- Find slow steps (is the kitchen slow?)
- Track costs (are we profitable?)
- Improve service (what do customers like?)
"""
