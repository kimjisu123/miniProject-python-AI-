from sqlalchemy import Column, Integer, Boolean, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Member(Base):
    __tablename__ = 'member'
    member_id = Column(Integer, primary_key=True) 
    member_name = Column(Text)
    member_phone = Column(Text)
    member_image = Column(LargeBinary)