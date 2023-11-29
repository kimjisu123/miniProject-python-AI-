from sqlalchemy import Column, Integer, Boolean, Text, DateTime
from database import Base

class Attendance(Base):
    __tablename__ = 'attendance'
    attendance_id = Column(Integer, primary_key=True) 
    start = Column(DateTime)
    end = Column(DateTime)