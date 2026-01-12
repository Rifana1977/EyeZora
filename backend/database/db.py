from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["ai_exam"]

violations_collection = db["violations"]
questions_collection = db["questions"]
exams_collection = db["exams"]
