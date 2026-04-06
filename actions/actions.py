actions.py

# actions.py (place this file in your project root)
import logging
import os
import requests  # <-- ADDED
import json      # <-- ADDED
from dotenv import load_dotenv
from typing import Text, Dict, Any, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Loading actions module (actions.py)")

# Load environment variables from .env in the project root
load_dotenv()

# --- simple config (use env var or paste key if needed) ---
OR_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OR_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set. Please add it to your .env file or environment.")
MODEL_NAME = "google/gemma-2-9b-it:free"

# --- ADDED: API Endpoint and System Prompt ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are Aura, a supportive and empathetic AI assistant for mental well-being.
Your purpose is to listen, provide comfort, and offer a safe space for users to express their feelings.
Respond with kindness and understanding. Do not give medical advice.
Keep your responses conversational and not overly long.
"""

# helper (kept minimal to avoid errors)
def build_short_history(tracker: Tracker, max_turns: int = 8) -> str:
    events = tracker.events[::-1]
    history = []
    turns = 0
    for e in events:
        if turns >= max_turns:
            break
        if e.get("event") == "user" and e.get("text"):
            history.append(f"User: {e.get('text')}")
            turns += 1
        elif e.get("event") == "bot" and e.get("text"):
            history.append(f"Bot: {e.get('text')}")
    history = history[::-1]
    return "\n".join(history)

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

        # --- THIS BLOCK IS REPLACED ---
        # reply = f"(LLM) I heard: {user_message[:240]}"
        # --- START OF NEW API CALL BLOCK ---

        logger.info(f"Received user message: {user_message}")

        headers = {
            "Authorization": f"Bearer {OR_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Build the message history for the API
        # This is a simple version. For long conversations, you might want
        # to parse the build_short_history() string back into a list.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            # You could add more history here from the tracker
            {"role": "user", "content": user_message}
        ]
        
        data = {
            "model": MODEL_NAME,
            "messages": messages
        }

        reply = "I'm sorry, I'm having a little trouble thinking right now." # Default error message

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                data=json.dumps(data),
                timeout=20 # 20-second timeout
            )
            
            # Raise an error if the API call was unsuccessful
            response.raise_for_status() 
            
            json_response = response.json()
            
            if json_response.get("choices") and len(json_response["choices"]) > 0:
                # Extract the LLM's response text
                reply = json_response["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API returned no choices or unexpected format: {json_response}")
                reply = "I'm not sure what to say to that."

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API call failed: {e}")
            reply = "I'm having trouble connecting to my brain right now. Please try again in a moment."
        
        # --- END OF NEW API CALL BLOCK ---

        dispatcher.utter_message(text=reply)

        # update free_talk slot if present
        prev_hist = tracker.get_slot("free_talk_history") or ""
        # Now, the 'reply' variable contains the actual LLM response
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