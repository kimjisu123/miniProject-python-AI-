from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DB_URL = 'sqlite:///project.sqlite3' 

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})

Base = declarative_base()
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        # 마지막에 무조건 닫음
        db.close()

Base.metadata.create_all(bind = engine)
