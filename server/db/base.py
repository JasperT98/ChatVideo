from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from configs import SQLALCHEMY_DATABASE_URI
import json

# create_engine() 函数创建一个数据库引擎，该函数接受一个数据库 URI 和一个 JSON 序列化器作为参数
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

# sessionmaker() 函数创建一个会话工厂，该函数接受一个引擎对象和一些可选参数作为参数
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# declarative_base() 函数创建一个基类，该类用于定义 ORM 模型
Base = declarative_base()
