# actions.py (UPDATED FOR GEMINI API)
import logging
import os
from dotenv import load_dotenv
from typing import Text, Dict, Any, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# --- NEW: Import the Google GenAI SDK ---
from google import genai
from google.genai import types
from google.genai.errors import APIError 
# --- Configuration ---
logger = logging.getLogger(__name__)

# Load environment variables from .env in the project root
load_dotenv()

# --- Use env var for the Gemini API key ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. Please add it to your .env file or environment.")

# 2. Use a fast, free-tier model
MODEL_NAME = "gemini-2.5-flash"

# --- Client Setup ---
try:
    # Initialize the client using the API key
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}")

SYSTEM_PROMPT = """You are Aura, a supportive and empathetic AI assistant for mental well-being.
Your purpose is to listen, provide comfort, and offer a safe space for users to express their feelings.
Respond with kindness and understanding. Do not give medical advice.
Keep your responses conversational and not overly long.
"""

# ... (helper function build_short_history remains the same) ...

class ActionLLMResponse(Action):
    def name(self) -> Text:
        return "action_llm_response"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text") or ""
        if not user_message.strip():
            dispatcher.utter_message(text="I didn't catch that — can you say that again?")
            return []

        logger.info(f"Received user message: {user_message}")
        
        # --- START OF NEW GEMINI API CALL BLOCK ---

        # 1. Format the conversation for the API
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(user_message)]
            )
        ]

        reply = "I'm sorry, I'm having a little trouble thinking right now."

        try:
            # Call the Gemini API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=256
                )
            )
            
            # Extract the text response
            reply = response.text
            
        except APIError as e:
            logger.error(f"Gemini API call failed (API Error): {e}")
            reply = "I'm having trouble connecting to my brain right now. The API failed."
        except Exception as e:
            logger.error(f"Gemini API call failed (General Error): {e}")
            reply = "I'm having trouble connecting to my brain right now. Please try again."
        
        # --- END OF GEMINI API CALL BLOCK ---

        dispatcher.utter_message(text=reply)

        # ... (update slots and return statement remains the same) ...
        prev_hist = tracker.get_slot("free_talk_history") or ""
        new_piece = f"U: {user_message}\nB: {reply}\n" 
        new_hist = (prev_hist + "\n" + new_piece).strip()
        if len(new_hist) > 3000:
            new_hist = new_hist[-3000:]
        return [SlotSet("free_talk", True), SlotSet("free_talk_history", new_hist)]

class ActionExitFreeTalk(Action):
    def name(self) -> Text:
        return "action_exit_free_talk"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Okay — I'll wrap up our chat. Take care.")
        return [SlotSet("free_talk", False)]