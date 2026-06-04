from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, ConfigDict
from passlib.context import CryptContext
from contextlib import asynccontextmanager
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
import uuid
import asyncio
import time
import jwt
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "cognicare_db")
JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGO = "HS256"
JWT_EXP_HOURS = 24
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
analyzer = SentimentIntensityAnalyzer()

client: Optional[AsyncIOMotorClient] = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    print(f"MongoDB connected: {DB_NAME}")
    yield
    if client:
        client.close()
        print("MongoDB disconnected")

app = FastAPI(title="CogniCare API", lifespan=lifespan)
router = APIRouter(prefix="/api")


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    role: str


class ChatMessageCreate(BaseModel):
    message: str


class ChatMessageResponse(BaseModel):
    message_id: str
    user_message: str
    ai_response: str
    response_time: float
    sentiment: str
    cognitive_load_score: float
    timestamp: str
    adapted_response: bool = False


class DailyMetrics(BaseModel):
    date: str
    avg_response_time: float
    interaction_efficiency_score: float
    avg_sentiment_score: float
    avg_cognitive_load_score: float
    message_count: int


class HeatmapData(BaseModel):
    date: str
    hour: int
    cognitive_load: float
    interaction_count: int


class CorrelationData(BaseModel):
    date: str
    avg_sentiment: float
    avg_response_time: float


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> User:
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await db.users.find_one({"id": user_id}, {"_id": 0, "password": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def sentiment_to_score(sentiment: str) -> float:
    if sentiment == "positive":
        return 1.0
    if sentiment == "negative":
        return -1.0
    return 0.0


async def analyze_metrics(message: str, response_time: float):
    compound = analyzer.polarity_scores(message)["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    time_factor = min(response_time / 10, 1.0)
    length_factor = 1.0 - min(len(message) / 200, 1.0)
    sentiment_factor = 0.7 if sentiment == "negative" else (0.3 if sentiment == "neutral" else 0.1)
    cognitive_load = (time_factor * 0.4 + length_factor * 0.3 + sentiment_factor * 0.3) * 10
    return sentiment, round(cognitive_load, 2)


class MockLlmChat:
    def __init__(self, system_message: str = "", history: Optional[list] = None):
        self.system_message = system_message
        self.history = history or []

    async def send_message(self, msg: str) -> str:
        await asyncio.sleep(0.4)
        msg_l = msg.lower()
        recent_messages = [m.lower() for m in self.history[-3:]]
        recent_stress = any(any(k in m for k in ["stress", "anxious", "panic", "overwhelmed"]) for m in recent_messages)

        if any(p in msg_l for p in ["i want to die", "kill myself", "suicide", "end my life"]):
            return random.choice([
                "I'm really sorry you're feeling this way. Please reach out to someone you trust or a local crisis helpline right now.",
                "You matter, and you do not have to face this alone. Please contact someone you trust or an emergency support line now.",
                "It sounds like you're in intense pain. Please seek immediate support from a trusted person or crisis service."
            ])
        elif recent_stress and any(w in msg_l for w in ["stress", "anxious", "panic", "overwhelmed"]):
            return "It sounds like this stress has been building for a while. Let's slow down and focus on one small step you can take right now."
        elif any(p in msg_l for p in ["no friends", "i am alone", "lonely", "isolated", "no one"]):
            return random.choice([
                "That sounds really lonely. I'm here with you—do you want to talk more about it?",
                "Feeling alone can be heavy. You are not alone in this moment.",
                "I understand. That must feel painful. Tell me a little more."
            ])
        elif any(p in msg_l for p in ["thank you", "thanks", "appreciate", "grateful"]):
            return random.choice([
                "You're always welcome. I'm here whenever you want to talk.",
                "I'm glad I could help. You can come back anytime.",
                "That means a lot. I'm here for you."
            ])
        elif any(w in msg_l for w in ["hi", "hello", "hey"]):
            return random.choice([
                "Hi. I'm here to listen—how are you feeling today?",
                "Hello. Tell me how your day has been.",
                "Hey, I'm here with you. What's on your mind?"
            ])
        elif any(w in msg_l for w in ["sad", "depressed", "low"]):
            return random.choice([
                "I'm sorry you're feeling this way. Want to share what's been hardest?",
                "That sounds tough. I'm here with you.",
                "You don't have to carry this alone."
            ])
        elif any(w in msg_l for w in ["stress", "anxious", "panic", "worry"]):
            return random.choice([
                "That sounds stressful. Let's take it one step at a time.",
                "I hear you. What is worrying you the most right now?",
                "It's okay to feel this way. Let's slow it down together."
            ])
        elif any(w in msg_l for w in ["tired", "exhausted", "burnt out", "sleepy", "no energy"]):
            return random.choice([
                "You sound exhausted. Have you been able to rest lately?",
                "It seems like you're really drained. A small break might help.",
                "That sounds tiring. Be gentle with yourself today."
            ])
        elif any(w in msg_l for w in ["exam", "study"]):
            return random.choice([
                "Exams can feel overwhelming. What subject are you working on?",
                "Let's break your studying into smaller steps.",
                "Study stress is real. What's feeling hardest right now?"
            ])
        elif any(w in msg_l for w in ["work", "job", "boss"]):
            return random.choice([
                "Work stress can be a lot. What happened?",
                "That sounds frustrating. Want to talk it through?"
            ])
        elif any(w in msg_l for w in ["breakup", "relationship", "love"]):
            return random.choice([
                "Relationships can be painful and confusing. I'm here for you.",
                "That must hurt. Do you want to share what happened?"
            ])
        elif any(w in msg_l for w in ["angry", "frustrated"]):
            return random.choice([
                "It's okay to feel angry. What led to that feeling?",
                "Let's slow down and talk through the frustration."
            ])
        elif any(w in msg_l for w in ["scared", "afraid"]):
            return random.choice([
                "That sounds scary. Do you want to talk about what's causing that fear?",
                "I'm here with you. Tell me what feels frightening right now."
            ])
        elif any(w in msg_l for w in ["confused", "lost"]):
            return random.choice([
                "It's okay to feel lost sometimes. What feels most confusing?",
                "Let's figure it out together, one part at a time."
            ])
        elif any(p in msg_l for p in ["what should i do", "help me", "advice"]):
            return random.choice([
                "Let's take it step by step. Tell me a bit more first.",
                "I'm here to help. What part feels most urgent?"
            ])
        elif "not" in msg_l and any(w in msg_l for w in ["good", "great", "happy", "well"]):
            return "I'm sorry things don't feel okay right now. Want to tell me more about what's going on?"
        elif any(w in msg_l for w in ["happy", "good", "great", "well", "awesome"]):
            return random.choice([
                "That's good to hear. What made you feel that way?",
                "I'm glad to hear that.",
                "That's a nice moment to hold onto."
            ])
        elif "simple" in self.system_message.lower():
            return "I understand. Let's keep it simple. Take a slow breath. You are doing okay."
        else:
            return random.choice([
                "Thank you for sharing that with me. I'm here to listen.",
                "I'm listening. Tell me a little more.",
                "It's okay to feel this way. I'm here with you."
            ])


@router.get("/")
async def root():
    return {"message": "CogniCare API is running"}


@router.get("/chat/welcome")
async def welcome_message():
    return ChatMessageResponse(
        message_id=str(uuid.uuid4()),
        user_message="",
        ai_response="Hi, it's good to see you — let's talk about how you're feeling today.",
        response_time=0.0,
        sentiment="neutral",
        cognitive_load_score=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        adapted_response=False,
    )


@router.post("/auth/register")
async def register(user_data: UserCreate):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "role": user_data.role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user_doc)

    return {
        "token": create_token({"user_id": user_id}),
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name,
            "role": user_data.role,
        },
    }


@router.post("/auth/login")
async def login(creds: UserLogin):
    user = await db.users.find_one({"email": creds.email})
    if not user or not verify_password(creds.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "token": create_token({"user_id": user["id"]}),
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        },
    }


@router.post("/chat/send", response_model=ChatMessageResponse)
async def send_message(msg_data: ChatMessageCreate, current_user: User = Depends(get_current_user)):
    start_time = time.time()

    previous_docs = await db.chat_messages.find(
        {"user_id": current_user.id}, {"_id": 0, "user_message": 1}
    ).sort("timestamp", 1).to_list(10)
    user_history = [doc.get("user_message", "") for doc in previous_docs if doc.get("user_message")]

    initial_sentiment, initial_load = await analyze_metrics(msg_data.message, 0)
    adapted_response = initial_load > 6.5
    system_message = (
        "You are a supportive mental health assistant. Keep responses short, clear, and calm."
        if adapted_response
        else "You are a thoughtful, empathetic mental health assistant."
    )

    chat = MockLlmChat(system_message=system_message, history=user_history)
    ai_response = await chat.send_message(msg_data.message)

    response_time = round(time.time() - start_time, 2)
    sentiment, cognitive_load_score = await analyze_metrics(msg_data.message, response_time)
    timestamp = datetime.now(timezone.utc).isoformat()
    message_id = str(uuid.uuid4())

    await db.chat_messages.insert_one({
        "id": message_id,
        "user_id": current_user.id,
        "user_message": msg_data.message,
        "ai_response": ai_response,
        "response_time": response_time,
        "sentiment": sentiment,
        "sentiment_score": sentiment_to_score(sentiment),
        "cognitive_load_score": cognitive_load_score,
        "timestamp": timestamp,
        "adapted_response": adapted_response,
    })

    return ChatMessageResponse(
        message_id=message_id,
        user_message=msg_data.message,
        ai_response=ai_response,
        response_time=response_time,
        sentiment=sentiment,
        cognitive_load_score=cognitive_load_score,
        timestamp=timestamp,
        adapted_response=adapted_response,
    )


@router.get("/chat/history")
async def get_chat_history(current_user: User = Depends(get_current_user), limit: int = Query(50, ge=1, le=200)):
    return await db.chat_messages.find(
        {"user_id": current_user.id}, {"_id": 0}
    ).sort("timestamp", -1).limit(limit).to_list(limit)


@router.get("/metrics/daily", response_model=list[DailyMetrics])
async def get_daily_metrics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    patient_id: Optional[str] = None,
):
    target_user_id = patient_id if (current_user.role == "clinician" and patient_id) else current_user.id
    messages = await db.chat_messages.find({"user_id": target_user_id}, {"_id": 0}).to_list(10000)

    daily_data = {}
    for msg in messages:
        try:
            msg_dt = datetime.fromisoformat(msg["timestamp"])
        except Exception:
            continue
        if msg_dt < datetime.now(timezone.utc) - timedelta(days=days):
            continue
        date_str = msg_dt.date().isoformat()
        daily_data.setdefault(date_str, {
            "response_times": [],
            "cognitive_loads": [],
            "sentiments": [],
            "message_count": 0,
        })
        daily_data[date_str]["response_times"].append(msg.get("response_time", 0.0))
        daily_data[date_str]["cognitive_loads"].append(msg.get("cognitive_load_score", 0.0))
        daily_data[date_str]["sentiments"].append(msg.get("sentiment_score", sentiment_to_score(msg.get("sentiment", "neutral"))))
        daily_data[date_str]["message_count"] += 1

    metrics = []
    for date_str, data in daily_data.items():
        avg_response_time = sum(data["response_times"]) / len(data["response_times"])
        avg_cognitive_load = sum(data["cognitive_loads"]) / len(data["cognitive_loads"])
        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        efficiency_score = max(0, 100 - (avg_response_time * 5) - (avg_cognitive_load * 5) + (avg_sentiment * 10))
        metrics.append(DailyMetrics(
            date=date_str,
            avg_response_time=round(avg_response_time, 2),
            interaction_efficiency_score=round(efficiency_score, 2),
            avg_sentiment_score=round(avg_sentiment, 2),
            avg_cognitive_load_score=round(avg_cognitive_load, 2),
            message_count=data["message_count"],
        ))
    return sorted(metrics, key=lambda x: x.date)


@router.get("/metrics/heatmap", response_model=list[HeatmapData])
async def get_heatmap_data(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    patient_id: Optional[str] = None,
):
    target_user_id = patient_id if (current_user.role == "clinician" and patient_id) else current_user.id
    messages = await db.chat_messages.find({"user_id": target_user_id}, {"_id": 0}).to_list(10000)

    heatmap = {}
    for msg in messages:
        try:
            msg_dt = datetime.fromisoformat(msg["timestamp"])
        except Exception:
            continue
        if msg_dt < datetime.now(timezone.utc) - timedelta(days=days):
            continue
        key = f"{msg_dt.date().isoformat()}_{msg_dt.hour}"
        heatmap.setdefault(key, {
            "date": msg_dt.date().isoformat(),
            "hour": msg_dt.hour,
            "loads": [],
            "count": 0,
        })
        heatmap[key]["loads"].append(msg.get("cognitive_load_score", 0.0))
        heatmap[key]["count"] += 1

    result = []
    for item in heatmap.values():
        avg_load = sum(item["loads"]) / len(item["loads"])
        result.append(HeatmapData(
            date=item["date"],
            hour=item["hour"],
            cognitive_load=round(avg_load, 2),
            interaction_count=item["count"],
        ))
    return sorted(result, key=lambda x: (x.date, x.hour))


@router.get("/metrics/correlation", response_model=list[CorrelationData])
async def get_correlation_data(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    patient_id: Optional[str] = None,
):
    target_user_id = patient_id if (current_user.role == "clinician" and patient_id) else current_user.id
    messages = await db.chat_messages.find({"user_id": target_user_id}, {"_id": 0}).to_list(10000)

    grouped = {}
    for msg in messages:
        try:
            msg_dt = datetime.fromisoformat(msg["timestamp"])
        except Exception:
            continue
        if msg_dt < datetime.now(timezone.utc) - timedelta(days=days):
            continue
        date_str = msg_dt.date().isoformat()
        grouped.setdefault(date_str, {"response_times": [], "sentiments": []})
        grouped[date_str]["response_times"].append(msg.get("response_time", 0.0))
        grouped[date_str]["sentiments"].append(msg.get("sentiment_score", sentiment_to_score(msg.get("sentiment", "neutral"))))

    result = []
    for date_str, data in grouped.items():
        result.append(CorrelationData(
            date=date_str,
            avg_sentiment=round(sum(data["sentiments"]) / len(data["sentiments"]), 2),
            avg_response_time=round(sum(data["response_times"]) / len(data["response_times"]), 2),
        ))
    return sorted(result, key=lambda x: x.date)


@router.get("/clinician/patients")
async def get_patients(current_user: User = Depends(get_current_user)):
    if current_user.role != "clinician":
        raise HTTPException(status_code=403, detail="Only clinicians can access this endpoint")
    return await db.users.find({"role": "patient"}, {"_id": 0, "password": 0}).to_list(1000)


app.include_router(router)
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
