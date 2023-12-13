from functools import wraps
from contextlib import contextmanager
from server.db.base import SessionLocal

'''
@contextmanager 是 Python 标准库 contextlib 模块中的一个装饰器，
用于将一个生成器函数转换成一个上下文管理器，使得我们可以使用 with 语句来管理一段代码块的上下文

这里我们定义了一个名为 session_scope 的生成器函数，并使用 @contextmanager 装饰器将其转换成了一个上下文管理器。
- session_scope 用于自动获取数据库会话并在使用后关闭会话
在上下文管理器中，我们可以执行一些前置操作和后置操作，而这些操作通常需要使用 try/finally 语句来确保它们的执行。
在 try 块中，我们使用 yield 语句将资源返回给调用者。
在 finally 块中关闭会话。
如果在使用会话时发生异常，则会话将回滚并重新引发异常。
'''
@contextmanager
def session_scope():
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = SessionLocal()
    try:
        # 使用 yield 语句将资源返回给调用者
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


'''
with_session 是一个装饰器函数，它接受一个函数作为参数，并返回一个新的函数 wrapper。
- wrapper 使用 with 语句创建一个数据库会话，然后调用原始函数并传递会话作为第一个参数。
- 如果原始函数执行成功，则提交会话并返回结果。
- 否则，它会回滚会话并重新引发异常。
这个装饰器函数可以用于确保数据库操作的原子性和一致性。

@wraps 是 Python 标准库 functools 模块中的一个装饰器，
它用于将被装饰函数的元信息（如函数名、文档字符串等）复制到装饰器函数中，
以便于在调用被装饰函数时，能够正确地显示被装饰函数的元信息

*args 和 **kwargs 是用于函数定义的特殊语法，用于处理不定数量的参数。
- *args 用于处理不定数量的位置参数, *args 会将传入函数的位置参数打包成一个元组（tuple）
- **kwargs 用于处理不定数量的关键字参数, **kwargs 则会将传入函数的关键字参数打包成一个字典（dictionary）
'''
def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper


def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> SessionLocal:
    db = SessionLocal()
    return db
