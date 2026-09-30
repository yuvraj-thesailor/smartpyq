import uuid #for 128 bits unique ids
from datetime import datetime #accurate realtime datetime
from sqlalchemy import ( #data types required
    Column, String, Integer, Float, Text, ForeignKey, 
    DateTime, Table, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship #to use relationship as python properties
from pgvector.sqlalchemy import Vector #for vector column in the table
from app.core.database import Base
from app.core.config import settings

# Many-to-Many junction table between Questions and Tags
question_tags = Table(
    "question_tags",
    Base.metadata,
    Column("question_id", UUID(as_uuid=True), ForeignKey("questions.question_id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", UUID(as_uuid=True), ForeignKey("tags.tag_id", ondelete="CASCADE"), primary_key=True)
)

class Branch(Base):
    __tablename__ = "branches"

    branch_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    branch_code = Column(String(50), unique=True, nullable=False, index=True) # e.g. "CSE", "ECE"
    branch_name = Column(String(255), nullable=False)                         # e.g. "Computer Science & Engineering"

    subjects = relationship("Subject", back_populates="branch", cascade="all, delete-orphan")


class Subject(Base):
    __tablename__ = "subjects"

    subject_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_code = Column(String(50), nullable=False, index=True)             # e.g. "CS8492"
    subject_name = Column(String(255), nullable=False)                        # e.g. "Database Management Systems"
    semester = Column(Integer, nullable=False)                                # e.g. 4
    branch_id = Column(UUID(as_uuid=True), ForeignKey("branches.branch_id"), nullable=False)

    branch = relationship("Branch", back_populates="subjects")
    papers = relationship("Paper", back_populates="subject", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"

    paper_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("subjects.subject_id"), nullable=False)
    exam_year = Column(Integer, nullable=False, index=True)                   # e.g. 2023
    exam_type = Column(String(50), nullable=False)                            # e.g. "End Semester", "Mid Term", "Supplementary"
    file_url = Column(String(500), nullable=False)                            # Path to original uploaded PDF / image
    page_count = Column(Integer, default=1)
    status = Column(String(50), default="processed")                          # "pending", "processing", "processed", "failed"
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    subject = relationship("Subject", back_populates="papers")
    questions = relationship("Question", back_populates="paper", cascade="all, delete-orphan")


class Question(Base):
    __tablename__ = "questions"

    question_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.paper_id"), nullable=False)
    parent_question_id = Column(UUID(as_uuid=True), ForeignKey("questions.question_id"), nullable=True) # For sub-questions like 1(a)
    
    question_number = Column(String(50), nullable=False)                      # e.g. "1", "1(a)", "Q3 OR"
    question_text = Column(Text, nullable=False)
    marks = Column(Integer, nullable=True)                                    # e.g. 2, 7, 10, 16
    unit_module = Column(Integer, nullable=True)                              # e.g. 1, 2, 3, 4, 5
    image_ref = Column(String(500), nullable=True)                            # Path to cropped diagram/equation fallback
    
    # 384-dimensional vector embedding for all-MiniLM-L6-v2
    embedding_vector = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    ocr_confidence = Column(Float, nullable=True)                             # 0.0 - 100.0
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="questions")
    sub_questions = relationship("Question", backref="parent_question", remote_side=[question_id])
    tags = relationship("Tag", secondary=question_tags, back_populates="questions")


class Tag(Base):
    __tablename__ = "tags"

    tag_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tag_name = Column(String(100), unique=True, nullable=False, index=True)  # e.g. "b-tree", "normalization"
    source = Column(String(50), default="auto")                               # "auto" or "verified"

    questions = relationship("Question", secondary=question_tags, back_populates="tags")
    
