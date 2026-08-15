"""
EduRAG管理学(河南)在线问答系统 - 用户认证服务
功能: 用户注册/登录、JWT令牌、邀请码兑换、用量管理
"""
import os
import hashlib
import datetime

import jwt
import pymysql
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from base import Config, logger


# FastAPI 的 Bearer Token 安全方案
security = HTTPBearer()


class AuthService:
    """用户认证服务类"""

    def __init__(self):
        self.conn = None
        self.cur = None
        self._connect()

    # ==================== 数据库连接管理 ====================

    def _connect(self):
        """建立MySQL连接"""
        try:
            self.conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
            )
            self.cur = self.conn.cursor()
            logger.info("[auth.AuthService._connect] MySQL连接成功")
        except Exception as e:
            logger.exception("[auth.AuthService._connect] MySQL连接失败")
            raise e

    def check_and_reconnect(self):
        """检查连接状态，断开则重连"""
        try:
            self.conn.ping()
        except Exception:
            logger.info("[auth.AuthService.check_and_reconnect] 连接已断开，正在重连...")
            self.close()
            self._connect()

    def close(self):
        """关闭游标和连接"""
        try:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # ==================== 密码加密 ====================

    @staticmethod
    def _hash_password(password: str) -> str:
        """使用 sha256 + 随机salt 对密码进行哈希
        返回格式: salt_hex:hash_hex
        """
        salt = os.urandom(16)
        hash_bytes = hashlib.sha256(salt + password.encode('utf-8')).digest()
        return salt.hex() + ':' + hash_bytes.hex()

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        """验证密码是否匹配
        stored 格式: salt_hex:hash_hex
        """
        try:
            salt_hex, hash_hex = stored.split(':')
            salt = bytes.fromhex(salt_hex)
            computed = hashlib.sha256(salt + password.encode('utf-8')).hexdigest()
            return computed == hash_hex
        except Exception:
            return False

    # ==================== JWT 令牌 ====================

    @staticmethod
    def _generate_token(user_id: int, username: str, role: str) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(
                hours=Config.JWT_EXPIRE_HOURS
            ),
        }
        token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')
        return token

    @staticmethod
    def _decode_token(token: str) -> dict:
        """解码并验证JWT令牌，失败抛异常"""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期，请重新登录")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效的令牌")

    # ==================== 用户注册 ====================

    def register(self, username: str, password: str) -> dict:
        """用户注册
        返回: {"code": 0/1, "msg": "...", "user_id": ...}
        """
        self.check_and_reconnect()
        try:
            # 检查用户名是否已存在
            self.cur.execute(
                "SELECT id FROM users WHERE username = %s",
                (username,)
            )
            if self.cur.fetchone():
                logger.info(f"[auth.AuthService.register] 用户名已存在: {username}")
                return {"code": 1, "msg": "用户名已存在"}

            # 插入新用户，默认额度从配置读取
            password_hash = self._hash_password(password)
            self.cur.execute(
                "INSERT INTO users (username, password_hash, role, total_quota, used_count) "
                "VALUES (%s, %s, %s, %s, %s)",
                (username, password_hash, 'user', Config.DEFAULT_QUOTA, 0)
            )
            self.conn.commit()
            user_id = self.cur.lastrowid
            logger.info(f"[auth.AuthService.register] 注册成功: {username}, id={user_id}")
            return {"code": 0, "msg": "注册成功", "user_id": user_id}
        except Exception as e:
            self.conn.rollback()
            logger.exception(f"[auth.AuthService.register] 注册异常: {username}")
            return {"code": 1, "msg": f"注册失败: {str(e)}"}

    # ==================== 用户登录 ====================

    def login(self, username: str, password: str) -> dict:
        """用户登录
        返回: {"code": 0/1, "msg": "...", "token": "..."}
        """
        self.check_and_reconnect()
        try:
            self.cur.execute(
                "SELECT id, username, password_hash, role FROM users WHERE username = %s",
                (username,)
            )
            user = self.cur.fetchone()
            if not user:
                logger.info(f"[auth.AuthService.login] 用户不存在: {username}")
                return {"code": 1, "msg": "用户名或密码错误"}

            if not self._verify_password(password, user['password_hash']):
                logger.info(f"[auth.AuthService.login] 密码错误: {username}")
                return {"code": 1, "msg": "用户名或密码错误"}

            token = self._generate_token(user['id'], user['username'], user['role'])
            logger.info(f"[auth.AuthService.login] 登录成功: {username}")
            return {"code": 0, "msg": "登录成功", "token": token}
        except Exception as e:
            logger.exception(f"[auth.AuthService.login] 登录异常: {username}")
            return {"code": 1, "msg": f"登录失败: {str(e)}"}

    # ==================== JWT 验证中间件 ====================

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> dict:
        """FastAPI 依赖注入：从请求头中提取并验证JWT令牌
        用法: current_user: dict = Depends(auth_service.get_current_user)
        返回: {"user_id": int, "username": str, "role": str}
        """
        token = credentials.credentials
        payload = self._decode_token(token)
        return {
            'user_id': payload['user_id'],
            'username': payload['username'],
            'role': payload.get('role', 'user'),
        }

    # ==================== 邀请码兑换 ====================

    def redeem_invite_code(self, user_id: int, code: str) -> dict:
        """邀请码兑换
        验证邀请码未使用 → 增加用户额度 → 标记邀请码已使用
        返回: {"code": 0/1, "msg": "..."}
        """
        self.check_and_reconnect()
        try:
            # 查询邀请码
            self.cur.execute(
                "SELECT id, code, quota, is_used FROM invite_codes WHERE code = %s",
                (code,)
            )
            invite = self.cur.fetchone()
            if not invite:
                logger.info(f"[auth.AuthService.redeem_invite_code] 邀请码不存在: {code}")
                return {"code": 1, "msg": "邀请码不存在"}

            if invite['is_used']:
                logger.info(f"[auth.AuthService.redeem_invite_code] 邀请码已使用: {code}")
                return {"code": 1, "msg": "邀请码已被使用"}

            # 增加用户额度
            self.cur.execute(
                "UPDATE users SET total_quota = total_quota + %s WHERE id = %s",
                (invite['quota'], user_id)
            )

            # 标记邀请码已使用
            self.cur.execute(
                "UPDATE invite_codes SET is_used = %s, used_by = %s, used_at = NOW() WHERE id = %s",
                (True, user_id, invite['id'])
            )

            self.conn.commit()
            logger.info(
                f"[auth.AuthService.redeem_invite_code] "
                f"用户{user_id}兑换邀请码{code}成功，增加{invite['quota']}次额度"
            )
            return {"code": 0, "msg": f"兑换成功，增加{invite['quota']}次使用额度"}
        except Exception as e:
            self.conn.rollback()
            logger.exception(f"[auth.AuthService.redeem_invite_code] 兑换异常: {code}")
            return {"code": 1, "msg": f"兑换失败: {str(e)}"}

    # ==================== 用量检查 ====================

    def check_usage(self, user_id: int) -> dict:
        """检查用户已用次数是否超过总额度
        返回: {"code": 0/1, "msg": "...", "total_quota": int, "used_count": int, "remaining": int}
        code=0 表示未超限，code=1 表示已超限
        """
        self.check_and_reconnect()
        try:
            self.cur.execute(
                "SELECT total_quota, used_count FROM users WHERE id = %s",
                (user_id,)
            )
            user = self.cur.fetchone()
            if not user:
                logger.info(f"[auth.AuthService.check_usage] 用户不存在: {user_id}")
                return {"code": 1, "msg": "用户不存在"}

            remaining = user['total_quota'] - user['used_count']
            if remaining <= 0:
                logger.info(
                    f"[auth.AuthService.check_usage] "
                    f"用户{user_id}额度已用完: {user['used_count']}/{user['total_quota']}"
                )
                return {
                    "code": 1,
                    "msg": "使用次数已用完，请联系管理员或使用邀请码增加额度",
                    "total_quota": user['total_quota'],
                    "used_count": user['used_count'],
                    "remaining": 0,
                }

            return {
                "code": 0,
                "msg": "额度充足",
                "total_quota": user['total_quota'],
                "used_count": user['used_count'],
                "remaining": remaining,
            }
        except Exception as e:
            logger.exception(f"[auth.AuthService.check_usage] 检查用量异常: user_id={user_id}")
            return {"code": 1, "msg": f"检查用量失败: {str(e)}"}

    # ==================== 用量增加 ====================

    def increment_usage(self, user_id: int) -> dict:
        """每次问答后 used_count + 1
        返回: {"code": 0/1, "msg": "..."}
        """
        self.check_and_reconnect()
        try:
            self.cur.execute(
                "UPDATE users SET used_count = used_count + 1 WHERE id = %s",
                (user_id,)
            )
            self.conn.commit()
            logger.info(f"[auth.AuthService.increment_usage] 用户{user_id}使用次数+1")
            return {"code": 0, "msg": "使用次数已更新"}
        except Exception as e:
            self.conn.rollback()
            logger.exception(f"[auth.AuthService.increment_usage] 更新用量异常: user_id={user_id}")
            return {"code": 1, "msg": f"更新用量失败: {str(e)}"}
