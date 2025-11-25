# ======================================================
# 🧠 DAY 4: TEACH-THE-TUTOR (DSA EDITION)
# 🚀 Features: LinkedList, Stack, Queue & Active Recall
# ======================================================

import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

print("\n" + "💻" * 50)
print("🚀 DSA TUTOR - DAY 4 TUTORIAL")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("💻" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📚 KNOWLEDGE BASE (DSA DATA)
# ======================================================

# NEW JSON FILE
CONTENT_FILE = "dsa_content.json" 

# 🆕 NEW DSA TOPICS
DEFAULT_CONTENT = [
    {
        "id": "linkedlist",
        "title": "Linked List",
        "summary": "A Linked List is a linear data structure where elements (nodes) are stored at non-contiguous memory locations. Each node contains data and a pointer/reference to the next node. Types include singly, doubly, and circular linked lists.",
        "sample_question": "What is the difference between an array and a linked list?"
    },
    {
        "id": "stack",
        "title": "Stack",
        "summary": "A Stack is a linear data structure that follows the LIFO (Last In, First Out) principle. Elements are inserted (push) and removed (pop) only from the top. It is used in recursion, expression evaluation, and backtracking.",
        "sample_question": "What is the LIFO principle and how does it apply to a stack?"
    },
    {
        "id": "queue",
        "title": "Queue",
        "summary": "A Queue is a linear data structure that follows the FIFO (First In, First Out) principle. Elements are added at the rear (enqueue) and removed from the front (dequeue). Used in scheduling, BFS, and buffering.",
        "sample_question": "How is a queue different from a stack?"
    }
]

def load_content():
    """
    📖 Checks if DSA JSON exists. 
    If NO: Generates it from DEFAULT_CONTENT.
    If YES: Loads it.
    """
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        
        if not os.path.exists(path):
            print(f"⚠️ {CONTENT_FILE} not found. Generating DSA data...")
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)
            print("✅ DSA content file created successfully.")
            
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        print(f"⚠️ Error managing content file: {e}")
        return []

# Load data immediately
COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class TutorState:
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    
    def set_topic(self, topic_id: str):
        topic = next((item for item in COURSE_CONTENT if item["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# ======================================================
# 🛠️ TUTOR TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata], 
    topic_id: Annotated[str, Field(description="The ID of the topic (linkedlist, stack, queue)")]
) -> str:
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id.lower())
    
    if success:
        return f"Topic set to {state.current_topic_data['title']}. Ask if they want to 'Learn', take a 'Quiz', or 'Teach back'."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata], 
    mode: Annotated[str, Field(description="Modes: learn, quiz, teach_back")]
) -> str:
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    
    agent_session = ctx.userdata.agent_session 
    
    if agent_session:
        if state.mode == "learn":
            agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
            instruction = f"Mode: LEARN. Explain: {state.current_topic_data['summary']}"
        elif state.mode == "quiz":
            agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
            instruction = f"Mode: QUIZ. Ask this: {state.current_topic_data['sample_question']}"
        elif state.mode == "teach_back":
            agent_session.tts.update_options(voice="en-US-ken", style="Promo")
            instruction = "Mode: TEACH_BACK. Ask the user to explain the topic."
        else:
            return "Invalid mode."
    else:
        instruction = "Voice switch error."

    print(f"🔄 SWITCHING MODE -> {state.mode.upper()}")
    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="User explanation during teach-back")]
) -> str:
    print(f"📝 EVALUATING EXPLANATION: {user_explanation}")
    return "Evaluate the explanation, score it out of 10, and correct mistakes."

# ======================================================
# 🧠 AGENT DEFINITION
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""
            You are a DSA Tutor helping students learn LinkedList, Stack, and Queue.
            
            📚 TOPICS: {topic_list}
            
            MODES:
            - LEARN (Matthew)
            - QUIZ (Alicia)
            - TEACH_BACK (Ken)

            Behavior:
            - Ask the user which topic they want.
            - Switch modes using set_learning_mode tool.
            - Evaluate explanations using evaluate_teaching.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💻" * 25)
    print("🚀 STARTING DSA TUTOR SESSION")
    print(f"📚 Loaded {len(COURSE_CONTENT)} DSA topics")

    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    userdata.agent_session = session
    
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
