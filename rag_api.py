# 项目路径设置
import sys
import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, current_dir)

# todo 导入标准库：os 操作文件/目录，re 正则匹配，json 序列化，time 计时，uuid 生成唯一ID，asyncio 异步编程
import re, json, time, uuid, asyncio
# todo Optional：类型提示，表示某个参数可以是 None（可为空）
from typing import Optional
# todo BaseModel：Pydantic 的数据模型基类，用来定义请求/响应的数据格式，自动做校验
from pydantic import BaseModel
# todo WebSocketDisconnect：WebSocket 连接断开的异常类，用来捕获并处理客户端断连
from starlette.websockets import WebSocketDisconnect

# todo StaticFiles：FastAPI 的静态文件服务，用来托管前端 HTML/CSS/JS 文件
from fastapi.staticfiles import StaticFiles
# todo CORSMiddleware：跨域中间件，让前端网页可以跨域名访问后端 API
from fastapi.middleware.cors import CORSMiddleware
# todo FileResponse：返回一个文件作为 HTTP 响应（比如返回 index.html）
from fastapi.responses import FileResponse
# todo FastAPI：Web 框架核心类；WebSocket：WebSocket 连接支持；HTTPException：抛 HTTP 异常用
# todo Depends：FastAPI 依赖注入，用于认证中间件
from fastapi import FastAPI, WebSocket, HTTPException, Depends

from base import Config, logger
# todo IntegratedQASystem：混合问答系统（MySQL BM25 + RAG）
from main_plus import IntegratedQASystem
# todo AuthService：用户认证服务（注册/登录/JWT验证/邀请码/用量管理）
from auth import AuthService

# todo 创建应用实例
# todo FastAPI() 创建一个 Web 应用对象，title 和 description 会显示在自动生成的 API 文档页面上
app = FastAPI(
    title="问答系统API",
    description="集成MySQL和RAG的智能问答系统"
)

# 配置CORS跨域，允许前端访问
# todo CORS（跨域资源共享）：浏览器安全策略，默认禁止网页请求不同域名/端口的服务器
# todo 加了 CORSMiddleware 后，前端（如 localhost:3000）就能请求后端 API（如 localhost:8000）
app.add_middleware(
    CORSMiddleware, # 允许跨域请求
    allow_origins=["*"],  # 允许全部域名访问，在生产环境中应该限制为特定域名
    allow_credentials=True, # 允许使用Cookie及HTTP Authorization
    allow_methods=["*"], # 允许任意HTTP方法
    allow_headers=["*"], # 允许任意HTTP请求头
)

# 创建静态文件目录
os.makedirs("./static", exist_ok=True)

# 创建全局QA系统实例
# todo 全局单例：整个 Web 应用共用一个 IntegratedQASystem 实例，避免重复初始化数据库连接和模型
qa_system = IntegratedQASystem()

# 创建全局认证服务实例
# todo 全局单例：共用一个 AuthService 实例，管理用户注册/登录/JWT/用量
auth_service = AuthService()

# 定义日常问候用语模式和回复
# todo GREETING_PATTERNS：预定义的问候语列表，包含正则匹配模式和对应的回复模板
# todo 当用户发"你好"、"在吗"等问候语时，直接返回预设回复，不用走问答流程
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是EduRAG管理学智能答疑助手，专为河南专升本考生服务！有什么管理学问题可以帮你解答？"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是EduRAG管理学智能答疑助手，专注于河南专升本管理学考试的答疑解惑！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！随时为你解答管理学相关的问题，请问有什么可以帮你的？"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答管理学学习中的问题！有什么我可以帮你的？"
    }
]

# 定义请求模型
# todo QueryRequest：使用 Pydantic 定义 POST 请求体的数据格式
# todo FastAPI 会自动校验请求数据是否符合这个格式，不符合就返回 422 错误
class QueryRequest(BaseModel):
    query: str  # todo 用户的问题文本
    source_filter: Optional[str] = None  # todo 学科过滤（可选，None 表示不过滤）
    session_id: Optional[str] = None  # todo 会话 ID（可选，None 表示新会话）

# 定义响应模型
# todo QueryResponse：定义 HTTP 响应体的数据格式（用于自动生成 API 文档）
class QueryResponse(BaseModel):
    answer: str  # todo 系统生成的答案文本
    is_streaming: bool  # todo 是否为流式响应（True 表示答案还在生成中）
    session_id: str  # todo 当前会话 ID
    processing_time: float  # todo 处理耗时（秒）

# 定义认证相关请求模型
class RegisterRequest(BaseModel):
    username: str  # 用户名
    password: str  # 密码

class LoginRequest(BaseModel):
    username: str  # 用户名
    password: str  # 密码

class RedeemRequest(BaseModel):
    code: str  # 邀请码

# 添加静态文件服务
# todo mount：把 /static 路径映射到本地 static 目录，访问 /static/xxx 就会返回 static 目录下的文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径返回登录页
# todo @app.get("/")：装饰器，表示当用户访问网站根路径（如 http://localhost:8000/）时执行这个函数
# todo async def：异步函数，FastAPI 会用异步方式执行，提高并发性能
@app.get("/")
async def read_root():
    # todo FileResponse：把 static/login.html 文件返回给浏览器，未登录用户先进登录页
    logger.info("[rag_api.read_root] 访问根路径，返回登录页")
    return FileResponse("static/login.html")

# 创建新会话
# todo POST /api/create_session：前端调用这个接口获取一个新的 session_id
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# 查询历史消息
# todo GET /api/history/{session_id}：获取指定会话的对话历史
# todo {session_id} 是路径参数，FastAPI 自动从 URL 中提取出来
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system._get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        # todo HTTPException：主动抛出 HTTP 错误，FastAPI 会返回对应的状态码给前端
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

# 清除历史消息
# todo DELETE /api/history/{session_id}：删除指定会话的所有对话记录
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system._clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# ==================== 认证相关API ====================

# 用户注册
@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """用户注册接口"""
    logger.info(f"[rag_api.register] 用户注册请求: {request.username}")
    result = auth_service.register(request.username, request.password)
    return result

# 用户登录
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    logger.info(f"[rag_api.login] 用户登录请求: {request.username}")
    result = auth_service.login(request.username, request.password)
    return result

# 邀请码兑换（需要认证）
@app.post("/api/auth/redeem")
async def redeem(
    request: RedeemRequest,
    current_user: dict = Depends(auth_service.get_current_user)
):
    """邀请码兑换接口，需要JWT认证"""
    user_id = current_user['user_id']
    logger.info(f"[rag_api.redeem] 用户{user_id}兑换邀请码: {request.code}")
    result = auth_service.redeem_invite_code(user_id, request.code)
    return result

# 查看用量（需要认证）
@app.get("/api/auth/usage")
async def check_usage(
    current_user: dict = Depends(auth_service.get_current_user)
):
    """查看当前用户使用额度，需要JWT认证"""
    user_id = current_user['user_id']
    logger.info(f"[rag_api.check_usage] 查询用户{user_id}用量")
    result = auth_service.check_usage(user_id)
    return result

# 收费接口预留 - 创建支付订单
@app.post("/api/payment/create_order")
async def create_payment_order():
    """创建支付订单 - 功能预留"""
    logger.info("[rag_api.create_payment_order] 支付功能尚未开放")
    return {"code": 1, "msg": "功能开发中，敬请期待"}

# 收费接口预留 - 支付回调
@app.post("/api/payment/callback")
async def payment_callback():
    """支付回调 - 功能预留"""
    logger.info("[rag_api.payment_callback] 支付回调功能尚未开放")
    return {"code": 1, "msg": "功能开发中"}


# 检查是否为日常问候用语并返回模板回复
# todo check_greeting：判断用户输入是不是"你好""在吗"之类的问候语，是的话直接返回预设回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除 # 前缀
    for pattern_info in GREETING_PATTERNS:
        # todo re.match：从字符串开头匹配正则，re.IGNORECASE 表示忽略大小写
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# 非流式查询接口 【mysql FQA】
# todo POST /api/query：普通的 HTTP POST 查询接口，适合只需要 MySQL 快速匹配的场景
# todo 需要JWT认证，验证用户身份和额度
@app.post("/api/query")
async def query(
    request: QueryRequest,
    current_user: dict = Depends(auth_service.get_current_user)
):
    start_time = time.time()  # 记录开始时间
    user_id = current_user['user_id']
    logger.info(f"[rag_api.query:{user_id}] 收到查询请求: {request.query[:50]}")

    # 检查用户额度
    usage_result = auth_service.check_usage(user_id)
    if usage_result['code'] == 1:
        return {
            "answer": usage_result['msg'],
            "is_streaming": False,
            "session_id": request.session_id or "",
            "processing_time": time.time() - start_time
        }

    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 问候语不消耗额度，直接返回
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 执行 BM25 搜索
    # todo bm25_search.search：在 MySQL 题库里用 BM25 算法搜索，返回 (答案, 是否需要RAG)
    answer, need_rag = qa_system.bm25_search.search(
        request.query, threshold=Config.BM25_THRESHOLD)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        # todo MySQL 没找到可靠答案时，告诉前端改用 WebSocket 接口获得流式响应
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 问答完成，增加用量
    auth_service.increment_usage(user_id)
    logger.info(f"[rag_api.query:{user_id}] 查询完成，已增加用量")
    # 返回 MySQL 答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }

# 流式查询WebSocket接口
# todo @app.websocket：WebSocket 路由，前端建立的 WebSocket 连接通过这个函数处理
# todo WebSocket 和普通 HTTP 不同：连接建立后双方可以随时互发消息，适合流式输出场景
# todo 认证方式：WebSocket不支持Authorization请求头，通过URL查询参数传递token
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    # todo 从URL查询参数中提取JWT token（如 ws://xxx/api/stream?token=xxx）
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="缺少认证令牌")
        logger.info("[rag_api.websocket] 连接被拒绝: 缺少token参数")
        return

    # 验证JWT令牌
    try:
        payload = auth_service._decode_token(token)
        user_id = payload['user_id']
        username = payload['username']
    except Exception as e:
        await websocket.close(code=4001, reason=str(e))
        logger.info(f"[rag_api.websocket] 连接被拒绝: token验证失败 - {str(e)}")
        return

    # 检查用户额度
    usage_result = auth_service.check_usage(user_id)
    if usage_result['code'] == 1:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "error": usage_result['msg']
        })
        await websocket.close(code=4003, reason="额度不足")
        logger.info(f"[rag_api.websocket:{user_id}] 额度不足，拒绝连接")
        return

    # todo websocket.accept()：接受客户端的 WebSocket 连接请求，三次握手
    await websocket.accept()  # 接受 WebSocket 连接
    logger.info(f"[rag_api.websocket:{user_id}] WebSocket连接已建立, 用户: {username}")
    try:
        while True:
            # 接收客户端消息
            # todo receive_text()：等待客户端发消息过来（阻塞式）
            data = await websocket.receive_text()
            try:
                request_data = json.loads(data)
            except json.JSONDecodeError:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "error",
                        "error": "请求格式错误，请发送JSON格式"
                    })
                continue
            # 获取查询参数
            query = request_data.get("query")
            if not query or not query.strip():
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "error",
                        "error": "查询内容不能为空"
                    })
                continue
            query = query.strip()
            source_filter = request_data.get("source_filter")
            session_id = request_data.get(
                "session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            # todo websocket.client_state：检查 WebSocket 连接是否还活着，防止往已经断开连接发消息
            if websocket.client_state == websocket.client_state.CONNECTED:
                # todo send_json：自动把 Python 字典转成 JSON 字符串发给客户端
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复（问候语不消耗额度）
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                continue
            # 调用问答系统，流式处理查询
            collected_answer = ""
            # todo qa_system.query 返回生成器，逐个推送 (chunk, is_complete)
            for chunk, is_complete in qa_system.query(
                    query, source_filter=source_filter, session_id=session_id):
                collected_answer += chunk  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if chunk and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    # todo 每个 chunk 都是答案的一个片段，前端逐字显示，实现打字机效果
                    await websocket.send_json({
                        "type": "token",
                        "token": chunk,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送来源引用信息（文件名+页码）
                        sources = getattr(qa_system.rag_system, '_last_sources', [])
                        if sources:
                            await websocket.send_json({
                                "type": "sources",
                                "sources": sources,
                                "session_id": session_id
                            })
                        # 发送结束标志
                        # todo type: "end" 告诉前端答案已经输出完毕，可以停止等待了
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    # 问答完成，增加用量
                    auth_service.increment_usage(user_id)
                    logger.info(f"[rag_api.websocket:{user_id}] 查询完成，已增加用量")
                    break
                # todo asyncio.sleep(0.01)：暂停10毫秒，控制流式输出速度，让前端有时间渲染
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        logger.info(f"[rag_api.websocket:{user_id}] WebSocket断开: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        logger.error(f"[rag_api.websocket:{user_id}] WebSocket错误: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息（不暴露内部异常详情）
            await websocket.send_json({
                "type": "error",
                "error": "服务内部错误，请稍后重试"
            })
    finally:
        # todo finally：不管正常还是异常，最后都要尝试关闭 WebSocket 连接
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            logger.error(f"[rag_api.websocket] 关闭WebSocket时出错: {str(e)}")


# 健康检查端点
# todo GET /health：健康检查接口，监控系统用它判断服务是否还活着（返回 200 就说明正常）
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 获取有效的学科类别
# todo GET /api/sources：返回所有可选的学科类别列表，前端拿来做下拉菜单
@app.get("/api/sources")
async def get_sources():
    return {"sources": Config.VALID_SOURCES}

# todo 如果直接运行这个文件（不是被 import），启动 uvicorn 服务器
if __name__ == "__main__":
    # todo uvicorn 是 FastAPI 的 ASGI 服务器，host="0.0.0.0" 监听所有网卡，port=8000 端口
    # todo reload=False 不自动重载，workers=1 只用1个工作进程
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=False, workers=1)

    r"""
    cd C:\......\EduRAG项目根目录下
    C:\Users\xxx\Desktop\Stady\郑州AI2期\venv_rag\Scripts\python.exe -m uvicorn rag_api:app --host 0.0.0.0 --port 8000
    """