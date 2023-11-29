from sqlalchemy import Column, Integer, Boolean, Text, LargeBinary, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Member(Base):
    __tablename__ = 'member'
    member_id = Column(Integer, primary_key=True) 
    member_name = Column(Text)
    member_phone = Column(Text)
    member_image = Column(LargeBinary)

    def __repr__(self):
        return f"<Member(member_id={self.member_id}, member_name={self.member_name}, member_phone={self.member_phone})>"

class Attendance(Base):
    __tablename__ = 'attendance'
    attendance_id = Column(Integer, primary_key=True) 
    start = Column(DateTime)
    end = Column(DateTime)
    mem_id = Column(Integer, ForeignKey('member.member_id'))
    
    member = relationship("Member")

    def __repr__(self):
        return f"<Attendance(attendance_id={self.attendance_id}, start={self.start}, end={self.end}, mem_id={self.mem_id})>"

