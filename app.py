from __future__ import annotations
import os
import copy
import gradio as gr
from demo_security import make_security_middleware
from dotenv import load_dotenv
import http.client
import json
import re
import random
import uuid
from urllib.parse import urlparse
import prompts  # 动态读取当前语言与分类提示词
from typing import Any, List
from music_db import MusicDatabase
from memory_manager import MemoryManager
from gim_audio_agent import GIMAudioAgent  # 新增：导入GIM音频编辑代理
from baseline_retriever import retrieve_baseline_track
from kimusic_client import create_proxy_generated_track
from kimusic_renderer import render_from_session_data
from prompts import get_full_system_prompt, MUSIC_NOTE, MUSIC_SELECTION_INFORMATION_TEMPLATE
from prompts import get_music_system_prompt, get_music_user_prompt
from hyper_parameters import (
    NUM_MUSIC_TRACKS, 
    MAX_MEMORY_ITEMS, 
    MAX_CONVERSATION_LENGTH,
    SUMMARIZATION_THRESHOLD,
    MODEL_NAME,
    MAX_TOKENS_RESPONSE,
    MAX_TOKENS_SUMMARY,
    MOOD_OPTIONS_NUM,
    GENRE_OPTIONS_NUM,
    REBUILD_MUSIC_INDEX
)
import time
from datetime import datetime
from session_logger import (
    append_ablation_session_log,
    add_music_track,
    create_empty_session,
    log_message,
    prepare_session_metadata,
    save_llm_estimated_va,
    save_session_json,
    update_music_track_feedback,
)
from session_manager import (
    generate_participant_id,
    get_next_session_info,
    init_session as init_ablation_session,
    start_washout,
)

try:
    import gradio_client.utils as _gcu

    _orig_json2py = _gcu._json_schema_to_python_type

    def _json_schema_to_python_type_patched(schema, defs=None):
        # JSON Schema supports boolean schemas:
        # True  => any schema allowed
        # False => no schema allowed (we still treat as Any to avoid crashing)
        if schema is True or schema is False:
            return "Any"
        return _orig_json2py(schema, defs)

    _gcu._json_schema_to_python_type = _json_schema_to_python_type_patched
except Exception as _e:
    # If patch fails, we don't block app startup; it will just error as before.
    pass

# Load environment variables
load_dotenv()


def make_llm_connection(timeout=None):
    """Create a connection to an OpenAI-compatible chat-completions endpoint."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "https"
    host = parsed.netloc or parsed.path
    base_path = parsed.path.rstrip("/")
    api_path = f"{base_path}/chat/completions" if base_path else "/v1/chat/completions"
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    if timeout is None:
        return conn_cls(host), api_path
    return conn_cls(host, timeout=timeout), api_path

# 在文件顶部添加状态跟踪变量
PREVIOUS_STATE = None
CURRENT_STATE = None
# 全局变量，用于控制UI更新
SHOULD_UPDATE_MUSIC_UI = False

PANAS_ITEM_LABELS = {
    "interested": {"en": "Interested", "zh": "感兴趣"},
    "distressed": {"en": "Distressed", "zh": "心烦"},
    "excited": {"en": "Excited", "zh": "兴奋"},
    "upset": {"en": "Upset", "zh": "难受"},
    "strong": {"en": "Strong", "zh": "强而有力"},
    "guilty": {"en": "Guilty", "zh": "内疚"},
    "scared": {"en": "Scared", "zh": "害怕"},
    "hostile": {"en": "Hostile", "zh": "敌意"},
    "enthusiastic": {"en": "Enthusiastic", "zh": "热情"},
    "proud": {"en": "Proud", "zh": "自豪"},
    "irritable": {"en": "Irritable", "zh": "烦躁"},
    "alert": {"en": "Alert", "zh": "警觉"},
    "ashamed": {"en": "Ashamed", "zh": "羞愧"},
    "inspired": {"en": "Inspired", "zh": "受鼓舞"},
    "nervous": {"en": "Nervous", "zh": "紧张"},
    "determined": {"en": "Determined", "zh": "坚定"},
    "attentive": {"en": "Attentive", "zh": "专注"},
    "jittery": {"en": "Jittery", "zh": "坐立不安"},
    "active": {"en": "Active", "zh": "活跃"},
    "afraid": {"en": "Afraid", "zh": "恐惧"},
}

PANAS_POSITIVE_ITEMS = [
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active"
]

PANAS_NEGATIVE_ITEMS = [
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid"
]

PANAS_ITEM_KEYS = list(PANAS_ITEM_LABELS.keys())

SUS_ITEMS = [
    ("I think that I would like to use this system frequently", "我想我会经常使用这个系统"),
    ("I found the system unnecessarily complex", "我发现该系统过于复杂"),
    ("I thought the system was easy to use", "我认为该系统易于使用"),
    ("I think that I would need the support of a technical person to use this system", "我认为我需要技术人员的帮助才能使用这个系统"),
    ("I found the various functions in this system were well integrated", "我发现该系统能够很好地集成了各种功能"),
    ("I thought there was too much inconsistency in this system", "我认为该系统中存在大量的不一致"),
    ("I would imagine that most people would learn to use this system very quickly", "我想大多数用户能很快学会使用该系统"),
    ("I found the system very cumbersome to use", "我发现该系统使用起来很麻烦"),
    ("I felt very confident using the system", "我使用该系统时，感到很有信心"),
    ("I needed to learn a lot of things before I could get going with this system", "在使用该系统之前，我需要学习很多相关知识"),
]

THERAPY_EXPERIENCE_ITEMS = [
    ("T1_empathy", "AI therapist made me feel understood and supported", "AI 治疗师的对话让我感到被理解和支持"),
    ("T2_music_match", "The selected music matched my emotional state", "系统选择的音乐符合我当时的情绪状态"),
    ("T3_imagery_facilitation", "The music helped me generate meaningful imagery or associations", "音乐帮助我产生了有意义的画面或联想"),
    ("T4_interpretation_quality", "The AI's interpretation of my imagery made sense", "AI 对我意象的解读是有道理的"),
    ("T5_flow_comfort", "The pace and flow of the therapy felt comfortable", "整个治疗过程的节奏和流程让我感到舒适"),
    ("T6_reuse_intention", "I would use this system again for relaxation or self-exploration", "我愿意再次使用这个系统进行放松或自我探索"),
]


def get_ui_texts(is_chinese: bool):
    if is_chinese:
        return {
            "input_label": "分享您的想法...",
            "input_placeholder": "在此输入您的消息...",
            "submit": "发送",
            "finish_session": "结束对话",
            "sam_title_pre": "### 会前 SAM 评分",
            "sam_title_post": "### 会后 SAM 评分",
            "sam_title_default": "### SAM 评分",
            "sam_valence_label": "愉悦度（Valence）",
            "sam_valence_info": "您现在感觉是更愉快，还是更不愉快？",
            "sam_arousal_label": "唤醒度（Arousal）",
            "sam_arousal_info": "您现在是更平静放松，还是更兴奋紧张？",
            "sam_instruction": "Valence（愉悦度）: 1=非常不愉快, 5=中性, 9=非常愉快\n\nArousal（唤醒度）: 1=非常平静, 5=中等, 9=非常兴奋",
            "sam_submit": "提交 SAM",
            "sam_saved": "SAM 评分已保存。",
            "sam_not_needed": "当前无需评分。",
            "panas_title_pre": "### 会前 PANAS 评分",
            "panas_title_post": "### 会后 PANAS 评分",
            "panas_title_default": "### PANAS 评分",
            "panas_submit": "提交 PANAS",
            "panas_saved": "PANAS 评分已保存。",
            "panas_not_needed": "当前无需 PANAS 评分。",
            "sus_title": "### SUS 系统可用性量表",
            "sus_title_default": "### SUS 系统可用性量表",
            "sus_submit": "提交 SUS",
            "sus_saved": "SUS 已保存。",
            "sus_not_needed": "当前无需 SUS。",
            "therapy_title": "### 治疗体验专项评估",
            "therapy_title_default": "### 治疗体验专项评估",
            "therapy_submit": "提交体验评估",
            "therapy_saved": "治疗体验评估已保存。",
            "therapy_not_needed": "当前无需治疗体验评估。",
            "music_start": "我正在为您准备合适的音乐，请稍等片刻...",
            "music_processing": "我正在处理音乐，请稍等片刻。",
            "music_ready": "音乐已准备好，您可以开始聆听。",
            "music_experience": "接下来请先安静感受音乐，我会在结束后继续陪您。",
            "music_ended": "这段音乐已经结束。先慢慢回到当下，我们再一起回顾刚才的体验。",
            "post_music_reflection": "欢迎回来。先慢慢回到当下，不着急马上回答。\n可以先注意一下自己的呼吸、身体的感觉，或者此刻最明显的情绪。\n\n当您准备好的时候，可以慢慢回想一下，\n刚才的音乐体验里，有没有什么画面、情绪、想法或身体感觉让您印象特别深刻？",
            "closure": "我们已经接近这段疗愈体验的尾声了。\n如果您愿意，我们可以一起简单整理一下刚才的体验，然后完成最后的评估。",
            "post_session_assessment_intro": "在结束之前，我们会做一个简短的感受评估，\n帮助我们更好地理解这次体验对您的影响。\n请根据您此刻的感受进行选择，没有对错之分。",
            "play_music": "🎧 请使用下方播放器聆听音乐。",
            "participant_id_label": "参与者编号",
            "participant_id_placeholder": "系统会自动生成；第二次会话时可粘贴已保存的编号",
            "start_session": "开始实验",
            "participant_id_generated": "参与者编号已生成。第 {session_number}/2 次会话已准备好。请保存此编号以便完成第二次会话。",
            "all_sessions_complete": "所有指定会话均已完成。谢谢您。",
            "session_ready": "第 {session_number}/2 次会话已准备好。",
            "save_participant_id": "请保存您的参与者编号：\n\n**{participant_id}**\n\n您需要使用它完成第二次会话。",
            "washout_continue": "请先完成休息再进入第二次会话。剩余时间：{minutes}:{seconds:02d}。您的参与者编号是 {participant_id}。",
            "washout_title": "请在开始第二次会话前短暂休息 5 分钟。",
            "washout_id_intro": "您的参与者编号是：",
            "washout_instruction": "请保存此编号。倒计时结束后，第二次会话将可继续。",
            "washout_available_after": "可继续时间：{end_time}",
            "washout_calculating": "正在计算剩余时间...",
            "washout_done": "休息已完成。您可以点击“开始会话”继续第二次会话。",
            "washout_remaining": "剩余休息时间：",
            "washout_complete": "休息已完成。",
            "audio_label": "音乐播放器",
            "final_complete_title": "谢谢您，您已经完成两次会话。",
            "final_complete_body": "您的回答已保存。",
            "start_session_2": "开始第二次会话"
        }
    return {
        "input_label": "Share your thoughts...",
        "input_placeholder": "Type your message here...",
        "submit": "Send",
        "finish_session": "Finish Session",
        "sam_title_pre": "### Pre-session SAM Rating",
        "sam_title_post": "### Post-session SAM Rating",
        "sam_title_default": "### SAM Rating",
        "sam_valence_label": "Valence",
        "sam_valence_info": "How pleasant or unpleasant do you feel right now?",
        "sam_arousal_label": "Arousal",
        "sam_arousal_info": "How calm or excited do you feel right now?",
        "sam_instruction": "Valence: 1=Very unpleasant, 5=Neutral, 9=Very pleasant\n\nArousal: 1=Very calm, 5=Moderate, 9=Very excited",
        "sam_submit": "Submit SAM",
        "sam_saved": "SAM rating saved.",
        "sam_not_needed": "No SAM rating is required right now.",
        "panas_title_pre": "### Pre-session PANAS Rating",
        "panas_title_post": "### Post-session PANAS Rating",
        "panas_title_default": "### PANAS Rating",
        "panas_submit": "Submit PANAS",
        "panas_saved": "PANAS rating saved.",
        "panas_not_needed": "No PANAS rating is required right now.",
        "sus_title": "### System Usability Scale",
        "sus_title_default": "### System Usability Scale",
        "sus_submit": "Submit SUS",
        "sus_saved": "SUS saved.",
        "sus_not_needed": "No SUS rating is required right now.",
        "therapy_title": "### Therapy Experience",
        "therapy_title_default": "### Therapy Experience",
        "therapy_submit": "Submit Experience",
        "therapy_saved": "Therapy experience saved.",
        "therapy_not_needed": "No therapy experience rating is required right now.",
        "music_start": "I'm preparing suitable music for you, please wait a moment...",
        "music_processing": "I am processing the music now. Please wait a moment.",
        "music_ready": "The music is ready. You can start listening.",
        "music_experience": "Please take a moment to experience the music. I will continue with you afterward.",
        "music_ended": "The music has ended. Take a moment to return to the present, and then we can reflect on your experience together.",
        "post_music_reflection": "Welcome back. Take your time returning to the present moment.\nYou do not need to answer immediately. You can first notice your breath, your body, or the most noticeable feeling right now.\n\nWhen you feel ready, you can gently reflect on the music experience.\nWas there any image, emotion, thought, or bodily sensation that stood out to you?",
        "closure": "We are approaching the end of this therapeutic experience.\nIf you would like, we can briefly reflect on what you experienced and then complete the final assessment.",
        "post_session_assessment_intro": "Before we finish, we will do a brief assessment\nto better understand how this experience has affected you.\nPlease respond based on how you feel right now.\nThere are no right or wrong answers.",
        "play_music": "🎧 Please use the player below to listen to the music.",
        "participant_id_label": "Participant ID",
        "participant_id_placeholder": "Generated automatically, or paste saved ID for Session 2",
        "start_session": "Start Experiment",
        "participant_id_generated": "Participant ID generated. Session {session_number} of 2 is ready. Please save this ID for Session 2.",
        "all_sessions_complete": "All assigned sessions are complete. Thank you.",
        "session_ready": "Session {session_number} of 2 is ready.",
        "save_participant_id": "Please save your Participant ID:\n\n**{participant_id}**\n\nYou will need it to complete Session 2.",
        "washout_continue": "Please continue your break before Session 2. Remaining time: {minutes}:{seconds:02d}. Your Participant ID is {participant_id}.",
        "washout_title": "Please take a short 5-minute break before starting Session 2.",
        "washout_id_intro": "Your Participant ID is:",
        "washout_instruction": "Please save this ID. Session 2 will become available when the countdown finishes.",
        "washout_available_after": "Available after: {end_time}",
        "washout_calculating": "Calculating remaining time...",
        "washout_done": "The break is complete. You may click Start Session to continue to Session 2.",
        "washout_remaining": "Remaining break time:",
        "washout_complete": "Break complete.",
        "audio_label": "Music Player",
        "final_complete_title": "Thank you. You have completed both sessions.",
        "final_complete_body": "Your responses have been saved.",
        "start_session_2": "Start Session 2"
    }

class GIMState:
    PRELUDE = "prelude"
    INDUCTION = "induction"
    MUSIC_IMAGING = "music_imaging"
    POSTLUDE = "postlude"


SESSION_OVER_TOKEN = "<session_over>"
SESSION_OVER_PROMPT_INSTRUCTION = (
    "\n\nPOSTLUDE SESSION CONTROL:\n"
    "Append the control token <session_over> on a separate final line ONLY IF "
    "all conditions are satisfied: 1. The participant's experience has been "
    "summarized. 2. The experience has been connected back to the participant's "
    "original concerns or current life. 3. A clear closing / return-to-present "
    "statement has been given. 4. NO further reflective question is asked. "
    "If the assistant is still asking questions, inviting further exploration, "
    "or continuing the conversation, DO NOT emit <session_over>. Do not explain "
    "this token to the participant."
)
MUSIC_IMAGING_MIN_USER_TURNS = 2
MUSIC_IMAGING_MAX_USER_TURNS = 4
POSTLUDE_MAX_USER_TURNS = 3


def is_session_chinese(session):
    return getattr(session, "language", "en") == "zh"


class GIMTherapySession:
    def __init__(
        self,
        startup_light: bool = False,
        user_id: str = None,
        condition: str = None,
        condition_order: list = None,
        session_number: int = 1,
        session_id: str = None,
        language: str = "en",
    ):
        init_started_at = time.time()
        print(f"[startup] GIMTherapySession.__init__ start startup_light={startup_light}")
        self.startup_light = startup_light
        self.current_state = GIMState.PRELUDE
        self.chat_history = []
        self.timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.language = language or "en"
        import hyper_parameters
        default_condition = "kimusic" if hyper_parameters.USE_KIMUSIC_GENERATION else "baseline"
        self.user_id = user_id or generate_participant_id()
        self.condition = condition or default_condition
        self.condition_order = condition_order or [self.condition]
        self.session_number = session_number
        self.session_data = create_empty_session(user_id=self.user_id, condition=self.condition)
        if session_id:
            self.session_data["session_id"] = session_id
        self.session_data["condition_order"] = list(self.condition_order)
        self.session_data["session_number"] = self.session_number
        self.session_data["timestamp_start"] = self.timestamp_start
        self.focus_intention = None
        # Initialize with welcome message
        welcome_started_at = time.time()
        self.initialize_welcome_message()
        print(f"[startup] initialize_welcome_message completed in {time.time() - welcome_started_at:.3f}s")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        # 可以通过环境变量控制是否重建索
        music_db_started_at = time.time()
        print(f"[startup] MusicDatabase construction start use_elasticsearch={not startup_light}")
        self.music_db = MusicDatabase("../toy_dataset/music_data_complete_with_valence_arousal.json", use_elasticsearch=not startup_light, rebuild_index=True) if not startup_light else None
        if startup_light:
            print("[startup] MusicDatabase deferred (lazy)")
        else:
            print(f"[startup] MusicDatabase construction completed in {time.time() - music_db_started_at:.3f}s")
        self.user_mood = None
        self.music_selected = False
        memory_started_at = time.time()
        print("[startup] MemoryManager construction start")
        self.memory = MemoryManager(max_memory_items=MAX_MEMORY_ITEMS, max_conversation_length=MAX_CONVERSATION_LENGTH) if not startup_light else None
        if startup_light:
            print("[startup] MemoryManager deferred (lazy)")
        else:
            print(f"[startup] MemoryManager construction completed in {time.time() - memory_started_at:.3f}s")
        self.last_assistant_message = None
        self.summarization_threshold = SUMMARIZATION_THRESHOLD
        self.selected_music_tracks = []  # Store selected music tracks
        
        # 初始化进度显示相关属性
        self.progress_status = ""  # 进度状态文本
        self.phase_info = ""      # 阶段信息
        self.music_info = ""      # 音乐处理信息
        
        # 新增：初始化GIM音频编辑代理
        audio_agent_started_at = time.time()
        print("[startup] GIMAudioAgent construction start")
        self.audio_agent = GIMAudioAgent(
            music_db=self.music_db,
            api_key=self.api_key,
            music_root="../toy_dataset/mp3",  ##todo：全句music root和output dir
            output_dir="output"
        ) if not startup_light else None
        if startup_light:
            print("[startup] GIMAudioAgent deferred (lazy)")
        else:
            print(f"[startup] GIMAudioAgent construction completed in {time.time() - audio_agent_started_at:.3f}s")
        self.gim_program_result = None  # 存储最终的Program合成结果
        self.kimusic_render_result = None
        self.baseline_used_track_ids = set()
        self.ablation_logged = False
        self.washout_pending = False
        self.experiment_started = False
        self.sam_state = {
            "ratings": [],
            "pending_phase": "pre_session",
            "pre_session_done": False,
            "post_session_done": False
        }
        self.session_completed = False
        self.panas_state = {
            "panas": [],
            "pending_phase": None,
            "pre_session_done": False,
            "post_session_done": False,
            "current_order": [],
            "current_order_phase": None
        }
        self.ui_message_flags = {
            "music_generation_started": False,
            "music_processing_started": False,
            "music_ready": False,
            "music_experience": False,
            "music_ended": False,
            "post_music_reflection": False,
            "closure": False,
            "final_assessment_intro": False
        }
        self.postlude_reflection_prompt_index = None
        self.phase_user_turns = {
            GIMState.PRELUDE: 0,
            GIMState.INDUCTION: 0,
            GIMState.MUSIC_IMAGING: 0,
            GIMState.POSTLUDE: 0,
        }
        print(f"[startup] GIMTherapySession.__init__ completed in {time.time() - init_started_at:.3f}s")

    def render_kimusic_track_once(self, proxy_track, fallback_audio_path):
        if self.kimusic_render_result is not None:
            return self.kimusic_render_result

        render_payload = copy.deepcopy(self.session_data)
        render_payload["generated_music"] = copy.deepcopy(proxy_track.get("generated_music"))
        render_payload["music_sequence"] = [copy.deepcopy(proxy_track)]

        self.kimusic_render_result = render_from_session_data(
            render_payload,
            output_dir="output/kimusic_generated",
            output_stem=f"{self.session_data.get('session_id', 'session')}_track0"
        )

        generated_music = proxy_track.setdefault("generated_music", {})
        if self.kimusic_render_result.get("success"):
            audio_file = self.kimusic_render_result.get("mp3_path")
            audio_file_full_path = os.path.abspath(audio_file)
            generated_music["audio_file"] = audio_file
            proxy_track["file_path"] = audio_file
            proxy_track["full_path"] = audio_file_full_path
            proxy_track["filename"] = os.path.basename(audio_file_full_path)
        else:
            generated_music["render_error"] = self.kimusic_render_result.get("error")
            proxy_track["file_path"] = fallback_audio_path
            proxy_track["full_path"] = os.path.abspath(fallback_audio_path) if fallback_audio_path else None
            proxy_track["filename"] = os.path.basename(fallback_audio_path) if fallback_audio_path else None
            if fallback_audio_path and os.path.exists(fallback_audio_path):
                generated_music["audio_file"] = os.path.relpath(os.path.abspath(fallback_audio_path)).replace(os.sep, "/")
            else:
                generated_music["audio_file"] = None

        return self.kimusic_render_result

    def ensure_music_db(self):
        if self.music_db is None:
            music_db_started_at = time.time()
            print(f"[startup] MusicDatabase construction start use_elasticsearch={not self.startup_light}")
            self.music_db = MusicDatabase(
                "../toy_dataset/music_data_complete_with_valence_arousal.json",
                use_elasticsearch=not self.startup_light,
                rebuild_index=True
            )
            print(f"[startup] MusicDatabase construction completed in {time.time() - music_db_started_at:.3f}s")
        return self.music_db

    def ensure_memory_manager(self):
        if self.memory is None:
            memory_started_at = time.time()
            print("[startup] MemoryManager construction start")
            self.memory = MemoryManager(
                max_memory_items=MAX_MEMORY_ITEMS,
                max_conversation_length=MAX_CONVERSATION_LENGTH
            )
            print(f"[startup] MemoryManager construction completed in {time.time() - memory_started_at:.3f}s")
        return self.memory

    def ensure_audio_agent(self):
        if self.audio_agent is None:
            audio_agent_started_at = time.time()
            print("[startup] GIMAudioAgent construction start")
            self.audio_agent = GIMAudioAgent(
                music_db=self.ensure_music_db(),
                api_key=self.api_key,
                music_root="../toy_dataset/mp3",
                output_dir="output"
            )
            print(f"[startup] GIMAudioAgent construction completed in {time.time() - audio_agent_started_at:.3f}s")
        return self.audio_agent

    def has_llm_estimated_va(self):
        va_data = self.session_data.get("llm_estimated_va", {})
        return va_data.get("current_state") is not None and va_data.get("target_state") is not None

    def _extract_json_block(self, text: str):
        if not isinstance(text, str):
            return None
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else None

    def _clamp_va_value(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(-1.0, min(1.0, numeric))

    def _parse_va_payload(self, text: str):
        json_block = self._extract_json_block(text)
        if not json_block:
            return None, None

        try:
            payload = json.loads(json_block)
        except Exception:
            return None, None

        def normalize_va(va_obj):
            if not isinstance(va_obj, dict):
                return None
            valence = self._clamp_va_value(va_obj.get("valence"))
            arousal = self._clamp_va_value(va_obj.get("arousal"))
            if valence is None or arousal is None:
                return None
            return {
                "valence": valence,
                "arousal": arousal
            }

        return normalize_va(payload.get("current_state_va")), normalize_va(payload.get("target_state_va"))

    def extract_va_from_prelude(self, user_message):
        if self.has_llm_estimated_va():
            return

        if self.current_state != GIMState.PRELUDE:
            return

        va_system_prompt = (
            prompts.PRELUDE_VA_EXTRACTION_PROMPT_ZH
            if getattr(prompts, "LANGUAGE", "en") == "zh"
            else prompts.PRELUDE_VA_EXTRACTION_PROMPT_EN
        )

        va_messages = [
            {
                "role": "system",
                "content": va_system_prompt
            },
            {
                "role": "user",
                "content": f"Conversation history: {self.chat_history[-8:] if len(self.chat_history) >= 8 else self.chat_history}\n\nUser's latest input: {user_message}"
            }
        ]

        conn, api_path = make_llm_connection()
        payload = json.dumps({
            "model": MODEL_NAME,
            "max_tokens": 120,
            "temperature": 0.1,
            "messages": va_messages
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            conn.request("POST", api_path, payload, headers)
            response = conn.getresponse()
            response_data = json.loads(response.read().decode("utf-8"))

            va_text = None
            if isinstance(response_data, dict):
                if 'choices' in response_data and response_data['choices']:
                    va_text = response_data['choices'][0]['message'].get('content')
                elif 'content' in response_data and response_data['content']:
                    va_text = response_data['content'][0].get('text')

            current_va, target_va = self._parse_va_payload(va_text)
            if current_va is not None and target_va is not None:
                save_llm_estimated_va(self.session_data, current_va, target_va)
                print(f"[va] extracted current={current_va} target={target_va}")
        except Exception as e:
            print(f"[va] extraction failed: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_logging_phase(self):
        phase = getattr(self, "current_state", None)
        if phase in (GIMState.PRELUDE, GIMState.INDUCTION, GIMState.MUSIC_IMAGING, GIMState.POSTLUDE):
            return phase
        return "system"

    def _normalize_message_content(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            if isinstance(content.get("content"), str):
                return content["content"]
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        if isinstance(content, (list, tuple)):
            parts = []
            for item in content:
                normalized_item = self._normalize_message_content(item)
                if normalized_item:
                    parts.append(normalized_item)
            return " ".join(parts)
        if content is None:
            return ""
        return str(content)

    def append_chat_message(self, role: str, content, phase: str = None):
        normalized_content = self._normalize_message_content(content)
        message = {"role": role, "content": normalized_content}
        self.chat_history.append(message)
        log_message(self.session_data, role, normalized_content, phase or self.get_logging_phase())
        return message

    def strip_session_over_token(self, text: str):
        if not isinstance(text, str):
            return text
        return text.replace(SESSION_OVER_TOKEN, "").strip()

    def response_still_invites_continuation(self, text: str):
        if not isinstance(text, str):
            return False
        normalized = text.lower()
        continuation_markers = [
            "?",
            "？",
            "什么感觉",
            "有什么变化",
            "你注意到",
            "你愿意",
            "你可以慢慢感受",
            "浠€涔堟劅瑙",
            "鏈変粈涔堝彉鍖",
            "浣犳敞鎰忓埌",
            "浣犳効鎰",
            "浣犲彲浠ユ參鎱㈡劅鍙",
            "what do you notice",
            "how does it feel",
            "what happens",
            "would you like",
        ]
        return any(marker in normalized for marker in continuation_markers)

    def mark_ready_for_post_session_assessment(self):
        self.session_completed = True
        self.sam_state["pending_phase"] = "post_session"
        self.panas_state["pending_phase"] = None
        self.panas_state["current_order"] = []
        self.panas_state["current_order_phase"] = None

    def append_music_tracks_to_session_data(self):
        import hyper_parameters

        prepare_session_metadata(
            self.session_data,
            n_tracks=max(4, len(self.selected_music_tracks or []))
        )
        existing_indexes = {
            entry.get("track_index") for entry in self.session_data.get("music_sequence", [])
        }
        for idx, track in enumerate(self.selected_music_tracks or [], start=1):
            if idx in existing_indexes:
                continue
            if self.condition == "kimusic" and track.get("source") != "generated":
                waypoint_sequence = self.session_data.get("waypoint_sequence") or []
                waypoint_va = (
                    waypoint_sequence[min(idx - 1, len(waypoint_sequence) - 1)]
                    if waypoint_sequence
                    else self.session_data.get("target_state_va") or self.session_data.get("current_state_va")
                )
                fallback_audio_path = track.get("full_path") or track.get("file_path") or track.get("filename")
                proxy_track = create_proxy_generated_track(
                    waypoint_va=waypoint_va,
                    session_id=self.session_data.get("session_id"),
                    track_index=idx - 1,
                    fallback_audio_path=fallback_audio_path,
                    waypoint_sequence=waypoint_sequence
                )
                proxy_track["duration_seconds"] = track.get("duration_seconds")
                track = {**track, **proxy_track}
            add_music_track(self.session_data, idx, track)

    def update_latest_track_feedback_from_reflection(self, imagery_text=None):
        music_sequence = self.session_data.get("music_sequence", [])
        if not music_sequence:
            return

        track_index = music_sequence[-1].get("track_index")
        latest_music_sam = None
        for rating in reversed(self.sam_state.get("ratings", [])):
            if rating.get("phase") not in ("pre_session", "post_session"):
                latest_music_sam = rating
                break

        update_music_track_feedback(
            self.session_data,
            track_index,
            latest_music_sam.get("valence") if latest_music_sam else None,
            latest_music_sam.get("arousal") if latest_music_sam else None,
            None,
            imagery_text if imagery_text else None
        )
        
    def initialize_welcome_message(self):
        """Initialize the session with a welcome message"""
        print("[startup] initialize_welcome_message start (static welcome text)")
        welcome_message_en = """Hello! I'm your GIM (Guided Imagery and Music) therapy assistant. I'm here to guide you through a therapeutic journey combining music and imagery. 

I specialize in:
• Creating a safe and supportive environment for emotional exploration
• Using music to facilitate deep personal insights
• Guiding you through different phases of the GIM experience

Feel free to share whatever comes to your mind. Would you like to intriduce yourself and tell me what brings you here today?"""

        welcome_message_zh = """您好！我是您的GIM（引导式音乐与意象）治疗助手。我将陪伴您展开一段结合音乐与意象的治疗之旅。

我的专长包括：
• 创造安全和支持的情感探索环境
• 运用音乐促进深层的个人洞察
• 引导您经历GIM体验的不同阶段

请随意分享任何想法。您愿意先介绍一下自己并且告诉我是什么原因让您来到这里吗？"""

        welcome_message = welcome_message_zh if is_session_chinese(self) else welcome_message_en
        self.append_chat_message("assistant", welcome_message, phase=GIMState.PRELUDE)

    def ensure_sam_pending_phase(self):
        """Keep the lightweight SAM prompt state synchronized with the session phase."""
        pending_phase = self.sam_state.get("pending_phase")
        if not self.sam_state.get("pre_session_done") and pending_phase is None:
            self.sam_state["pending_phase"] = "pre_session"
            return

    def ensure_panas_pending_phase(self):
        """Keep the lightweight PANAS prompt state synchronized with phase and SAM completion."""
        pending_phase = self.panas_state.get("pending_phase")

        if (
            self.sam_state.get("pre_session_done")
            and not self.panas_state.get("pre_session_done")
            and pending_phase is None
        ):
            self.panas_state["pending_phase"] = "pre_session"

        if (
            self.sam_state.get("post_session_done")
            and not self.panas_state.get("post_session_done")
            and pending_phase is None
        ):
            self.panas_state["pending_phase"] = "post_session"

        phase = self.panas_state.get("pending_phase")
        if phase in ("pre_session", "post_session"):
            if self.panas_state.get("current_order_phase") != phase or not self.panas_state.get("current_order"):
                shuffled_keys = list(PANAS_ITEM_KEYS)
                random.shuffle(shuffled_keys)
                self.panas_state["current_order"] = shuffled_keys
                self.panas_state["current_order_phase"] = phase

    def get_panas_item_label(self, item_key: str, is_chinese: bool) -> str:
        label_key = "zh" if is_chinese else "en"
        return PANAS_ITEM_LABELS.get(item_key, {}).get(label_key, item_key)

    def build_session_results_export(self):
        """Build the unified session export payload."""
        self.session_data["timestamp_start"] = getattr(self, "timestamp_start", self.session_data.get("timestamp_start"))
        self.session_data["user_id"] = self.user_id
        self.session_data["condition"] = self.condition
        self.session_data["condition_order"] = list(self.condition_order or [])
        self.session_data["session_number"] = self.session_number
        prepare_session_metadata(
            self.session_data,
            n_tracks=max(4, len(self.session_data.get("music_sequence", []))),
            focus_theme=self.get_focus_theme_metadata()
        )
        return self.session_data

    def append_ablation_log_once(self):
        if self.ablation_logged:
            return False
        self.build_session_results_export()
        self.session_data["timestamp_end"] = self.session_data.get("timestamp_end") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logged = append_ablation_session_log(
            self.session_data,
            condition_order=self.condition_order,
            session_number=self.session_number,
            positive_items=PANAS_POSITIVE_ITEMS,
            negative_items=PANAS_NEGATIVE_ITEMS,
        )
        if logged:
            self.ablation_logged = True
        return logged

    def get_focus_theme_metadata(self):
        if self.focus_intention and self.focus_intention != "exploring inner experiences":
            return self.focus_intention

        if self.memory and self.memory.memories.get("focus_intentions"):
            latest_focus = self.memory.memories["focus_intentions"][-1]
            if isinstance(latest_focus, dict):
                return latest_focus.get("focus")
            return latest_focus

        return None

    def append_guidance_message_once(self, flag_key: str, message: str):
        """Append a single assistant guidance message once per session transition."""
        if self.ui_message_flags.get(flag_key):
            return False
        self.append_chat_message("assistant", message)
        self.ui_message_flags[flag_key] = True
        return True

    def has_postlude_reflection_response(self):
        """Check whether the user has responded after the post-music reflection prompt."""
        if self.postlude_reflection_prompt_index is None:
            return False
        for msg in self.chat_history[self.postlude_reflection_prompt_index + 1:]:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg["content"].strip():
                return True
        return False

    def has_postlude_integration_completed(self):
        """Check whether the postlude has had enough natural back-and-forth before final assessment."""
        if self.postlude_reflection_prompt_index is None:
            return False

        messages_after_postlude_start = self.chat_history[self.postlude_reflection_prompt_index + 1:]
        user_turns = 0
        assistant_turns = 0

        for msg in messages_after_postlude_start:
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if msg.get("role") == "user":
                user_turns += 1
            elif msg.get("role") == "assistant":
                assistant_turns += 1

        # Keep the original post-music exploration rhythm by waiting for at least
        # two user turns and two assistant turns in POSTLUDE before showing the
        # final post-session assessments.
        return user_turns >= 2 and assistant_turns >= 2

    def maybe_append_post_session_assessment_intro(self):
        """Preserve the original postlude flow without auto-triggering final assessments."""
        return False

    def get_clean_chat_history_for_model(self):
        """Filter invalid chat turns before sending history back to the model."""
        cleaned_messages = []
        for msg in self.chat_history:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant"):
                continue
            if not isinstance(content, str):
                continue

            stripped_content = content.strip()
            if not stripped_content:
                continue
            if stripped_content.startswith("API请求失败"):
                continue

            cleaned_messages.append({
                "role": role,
                "content": stripped_content
            })

        return cleaned_messages

    def get_system_prompt(self):
        """Get the system prompt based on current state, with memory information."""
        # Get memory information to add to the prompt
        self.ensure_memory_manager()
        memory_prompt = self.memory.get_memory_for_prompt(self.current_state)
            
        # Use the prompt builder to get the full system prompt
        system_prompt = get_full_system_prompt(self.current_state, memory_prompt)
        if self.current_state == GIMState.POSTLUDE:
            system_prompt += SESSION_OVER_PROMPT_INSTRUCTION
        return system_prompt

    def get_music_recommendation_prompts(self):
        """
        Generate a prompt for the LLM to provide music selection criteria.
        This is called when transitioning to the music_imaging phase.
        """
        
        # 获取焦点
        self.ensure_memory_manager()
        self.ensure_music_db()
        if not self.focus_intention:
            # Try to get focus intention from memory
            if self.memory.memories["focus_intentions"]:
                self.focus_intention = self.memory.memories["focus_intentions"][-1]["focus"]
            else:
                self.focus_intention = "exploring inner experiences"
        # 获取情绪
        if not self.user_mood:
            if self.memory.memories["emotional_states"]:
                emotions = [state["emotion"] for state in self.memory.memories["emotional_states"][-3:]]
                self.user_mood = "expressing " + ", ".join(emotions)
            else:
                # Try to infer mood from conversation history
                user_inputs = [msg["content"] for msg in self.chat_history if msg["role"] == "user"]
                if user_inputs:
                    last_message = user_inputs[-1]
                    self.user_mood = "expressing " + last_message[:30] + "..."
                else:
                    self.user_mood = "open to exploration"
        
        # Get music preferences from memory
        music_prefs = []
        if self.memory.memories["preferences"]["music"]:
            likes = [pref["genre"] for pref in self.memory.memories["preferences"]["music"] 
                    if pref["sentiment"] == "like"]
            if likes:
                music_prefs.append("likes " + ", ".join(likes))
                
            dislikes = [pref["genre"] for pref in self.memory.memories["preferences"]["music"] 
                       if pref["sentiment"] == "dislike"]
            if dislikes:
                music_prefs.append("dislikes " + ", ".join(dislikes))
                
        user_preferences = "; ".join(music_prefs) if music_prefs else None
        
        # Get mood and genre options
        mood_options = ", ".join(self.music_db.get_attribute_options("mood", MOOD_OPTIONS_NUM))
        genre_options = ", ".join(self.music_db.get_attribute_options("genre", GENRE_OPTIONS_NUM))
        # 生成prompt
        system_prompt = get_music_system_prompt(mood_options, genre_options)
        user_prompt = get_music_user_prompt(
            therapy_state=self.current_state,
            user_focus=self.focus_intention,
            user_mood=self.user_mood,
            user_preferences=user_preferences
        )
        return system_prompt, user_prompt

    def classify_state(self, user_message):
        """第一步：状态分类 - 仅返回状态标签，不生成正文内容"""
        # 使用prompts中新增的双语分类系统提示，基于当前语言动态选择
        classification_system_prompt = (
            prompts.STATE_CLASSIFICATION_SYSTEM_PROMPT_ZH
            if getattr(prompts, "LANGUAGE", "en") == "zh"
            else prompts.STATE_CLASSIFICATION_SYSTEM_PROMPT_EN
        )

        # 准备分类用的消息
        classification_messages = [
            {
                "role": "system",
                "content": classification_system_prompt
            }
        ]
        
        # 添加当前状态作为上下文
        classification_messages.append({
            "role": "user", 
            "content": f"Current state: {self.current_state}\n\nConversation history: {self.chat_history[-5:] if len(self.chat_history) >= 5 else self.chat_history}\n\nUser's latest input: {user_message}"
        })
        log_message(
            self.session_data,
            "assistant",
            f"State classification requested while in {self.current_state}.",
            "system"
        )
        
        # 发送分类请求
        conn, api_path = make_llm_connection()
        payload = json.dumps({
            "model": MODEL_NAME,
            "max_tokens": 32,  # 极轻量
            "temperature": 0.1,  # 低温度确保稳定
            "messages": classification_messages
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 发送API请求
        conn.request("POST", api_path, payload, headers)
        response = conn.getresponse()
        
        try:
            response_data = json.loads(response.read().decode("utf-8"))
            
            # 提取分类结果
            if isinstance(response_data, dict) and 'choices' in response_data:
                classification_text = response_data['choices'][0]['message']['content']
                
                # 提取状态
                state_match = re.search(r'<STATE>(.*?)</STATE>', classification_text)
                if state_match:
                    predicted_state = state_match.group(1)
                    print(f"状态分类结果：{predicted_state}")
                    return predicted_state
                else:
                    print(f"状态分类失败，未找到状态标签：{classification_text}")
                    return self.current_state  # 保持当前状态
            else:
                print(f"状态分类API响应格式异常：{response_data}")
                return self.current_state
                
        except Exception as e:
            print(f"状态分类过程中发生错误：{e}")
            return self.current_state

    # def get_next_response(self, user_message):
    #     """获取下一个响应 - 使用两步流水线：先判定状态，再生成内容"""
    #     global SHOULD_UPDATE_MUSIC_UI
        
    #     # 保存旧状态
    #     old_state = self.current_state
        
    #     # 第一步：状态分类
    #     print("classify state...")
    #     predicted_state = self.classify_state(user_message)
        
    #     # 更新状态
    #     self.current_state = predicted_state
    #     print("current state: ", self.current_state)
    #     # 处理状态变化
    #     if old_state != predicted_state:
    #         print(f"状态从 {old_state} 变为 {predicted_state}")
            
    #         # 如果状态从非音乐成像变为音乐成像，设置标志并选择音乐
    #         if old_state != GIMState.MUSIC_IMAGING and predicted_state == GIMState.MUSIC_IMAGING:
    #             print("状态变为音乐成像，设置UI更新标志并选择音乐")
    #             SHOULD_UPDATE_MUSIC_UI = True
    #             # 立即选择音乐
    #             if not self.music_selected:
    #                 self.select_music_for_imaging()
        
    #     # 处理用户消息（更新记忆）
    #     self.process_user_message(user_message)
        
    #     # 第二步：内容生成 - 使用新状态的system prompt
    #     response = self.get_ai_response()
        
    #     # 从响应中提取状态进行一致性校验
    #     extracted_state = self.extract_state_from_response(response)
    #     if extracted_state != predicted_state:
    #         print(f"警告：生成内容中的状态({extracted_state})与分类结果({predicted_state})不一致")
    #         # 可以选择更新状态或保持分类结果
    #         # self.current_state = extracted_state
        
    #     # 返回最终处理后的响应
    #     return self.last_assistant_message
    
    def get_next_response_stream(self, user_message):
        """获取下一个AI流式响应"""
        # 保存旧状态
        old_state = self.current_state

        # 第一步：状态分类
        print("classify state...")
        predicted_state = self.classify_state(user_message)
        if old_state == GIMState.PRELUDE and not self.has_llm_estimated_va():
            self.extract_va_from_prelude(user_message)

        # 记录当前 phase 的用户轮数：这是 hard cap，只限制最大轮次，不阻止 LLM 提前进入下一阶段
        if not hasattr(self, "phase_user_turns"):
            self.phase_user_turns = {
                GIMState.PRELUDE: 0,
                GIMState.INDUCTION: 0,
                GIMState.MUSIC_IMAGING: 0,
                GIMState.POSTLUDE: 0,
            }

        self.phase_user_turns[old_state] = self.phase_user_turns.get(old_state, 0) + 1

        # 最大轮数保险：如果 LLM 一直停留在 music_imaging，第 4 个用户回复后强制进入 postlude
        music_imaging_turns = self.phase_user_turns.get(GIMState.MUSIC_IMAGING, 0)
        if old_state == GIMState.MUSIC_IMAGING and music_imaging_turns < MUSIC_IMAGING_MIN_USER_TURNS:
            print("Music imaging minimum exploration not reached; staying in music_imaging.")
            predicted_state = GIMState.MUSIC_IMAGING
        elif old_state == GIMState.MUSIC_IMAGING and music_imaging_turns >= MUSIC_IMAGING_MAX_USER_TURNS:
            print("Music imaging turn limit reached; forcing transition to postlude.")
            predicted_state = GIMState.POSTLUDE
        
        # 更新状态
        self.current_state = predicted_state
        print("current state: ", self.current_state)
        
        # 处理状态变化
        if old_state != predicted_state:
            print(f"状态从 {old_state} 变为 {predicted_state}")
            
            # 如果状态从非音乐成像变为音乐成像，设置标志（但暂不立即选择音乐）
            if old_state != GIMState.MUSIC_IMAGING and predicted_state == GIMState.MUSIC_IMAGING:
                print("状态变为音乐成像，设置UI更新标志")
            elif old_state != GIMState.POSTLUDE and predicted_state == GIMState.POSTLUDE:
                self.postlude_reflection_prompt_index = max(-1, len(self.chat_history) - 2)
        
        # 处理用户消息（更新记忆）
        print("update memory...")
        start_time = time.time()
        self.process_user_message(user_message)
        end_time = time.time()
        print(f"update memory time: {end_time - start_time} seconds")
        
        # 第二步：内容生成 - 使用流式输出
        for chunk in self.get_ai_response_stream():
            yield chunk
        
        # # 从响应中提取状态进行一致性校验
        # response_text = self.last_assistant_message
        # extracted_state = self.extract_state_from_response(response_text)
        # if extracted_state != predicted_state:
        #     print(f"警告：生成内容中的状态({extracted_state})与分类结果({predicted_state})不一致")
    
    def update_program_progress(self, stage: str, status: str, progress: float, data: Any = None):
        """更新Program构建进度显示
        
        Args:
            stage: 当前阶段
            status: 状态信息
            progress: 进度值 (0-100)
            data: 相关数据
        """
        # 根据当前语言选择显示文本
        is_chinese = is_session_chinese(self)
        
        # 阶段名称映射
        stage_names = {
            "analysis": "分析" if is_chinese else "Analysis",
            "design": "设计" if is_chinese else "Design",
            "music_search": "音乐检索" if is_chinese else "Music Search",
            "processing": "音乐处理" if is_chinese else "Music Processing",
            "synthesis": "音频合成" if is_chinese else "Audio Synthesis"
        }
        
        # 更新进度状态
        progress_text = f"**{stage_names.get(stage, stage)}**: {status} ({progress:.0f}%)"
        self.progress_status = progress_text
        
        # 更新阶段信息
        if stage == "design" and data:
            if is_chinese:
                phase_text = "### 治疗阶段设计\n\n"
                for i, phase in enumerate(data.get("phases", []), 1):
                    phase_text += f"{i}. **{phase['name']}**\n"
                    phase_text += f"   - 时长: {phase['duration']}秒\n"
                    phase_text += f"   - 目的: {phase['purpose']}\n\n"
            else:
                phase_text = "### Therapy Phases Design\n\n"
                for i, phase in enumerate(data.get("phases", []), 1):
                    phase_text += f"{i}. **{phase['name']}**\n"
                    phase_text += f"   - Duration: {phase['duration']} seconds\n"
                    phase_text += f"   - Purpose: {phase['purpose']}\n\n"
            
            self.phase_info = phase_text
        
        # 更新音乐处理信息
        if stage == "processing" and data:
            if is_chinese:
                music_text = "### 音乐处理结果\n\n"
                for i, seg in enumerate(data.get("segments", []), 1):
                    music_text += f"{i}. **{seg['title']}**\n"
                    music_text += f"   - 时长: {seg['duration']}秒\n"
                    music_text += f"   - 处理参数:\n"
                    music_text += f"     - 速度: {seg['processing']['speed']}x\n"
                    music_text += f"     - 音调: {seg['processing']['pitch']} 半音\n"
                    music_text += f"     - 音量: {seg['processing']['volume']}x\n"
                    music_text += f"     - 淡入: {seg['processing']['fade_in']}ms\n"
                    music_text += f"     - 淡出: {seg['processing']['fade_out']}ms\n\n"
            else:
                music_text = "### Music Processing Results\n\n"
                for i, seg in enumerate(data.get("segments", []), 1):
                    music_text += f"{i}. **{seg['title']}**\n"
                    music_text += f"   - Duration: {seg['duration']} seconds\n"
                    music_text += f"   - Processing:\n"
                    music_text += f"     - Speed: {seg['processing']['speed']}x\n"
                    music_text += f"     - Pitch: {seg['processing']['pitch']} semitones\n"
                    music_text += f"     - Volume: {seg['processing']['volume']}x\n"
                    music_text += f"     - Fade in: {seg['processing']['fade_in']}ms\n"
                    music_text += f"     - Fade out: {seg['processing']['fade_out']}ms\n\n"
            
            self.music_info = music_text

    def select_music_for_imaging(self, progress_callback=None):
        """使用GIM音频编辑代理构建和渲染音乐Program"""
        # 如果已经选择过音乐，则不再重复选择
        if self.music_selected:
            print("Music already selected for this session, skipping selection")
            return

        if self.condition == "baseline":
            print("Selecting music through baseline retrieval path...")
            try:
                self.ensure_music_db()
                system_prompt, user_prompt = self.get_music_recommendation_prompts()
                criteria = self.music_db.get_music_criteria_json(system_prompt, user_prompt, self.api_key)
                baseline_track = retrieve_baseline_track(
                    self.music_db,
                    criteria,
                    track_index=0,
                    used_track_ids=self.baseline_used_track_ids,
                    music_root="../toy_dataset/mp3",
                )
                self.baseline_used_track_ids.add(baseline_track.get("track_id"))
                self.selected_music_tracks = [baseline_track]
                self.append_music_tracks_to_session_data()
                self.music_selected = True
                print(f"Baseline music selection completed - {baseline_track.get('track_id')}")
            except Exception as e:
                print(f"Baseline music selection failed: {e}")
                import traceback
                traceback.print_exc()
                self.music_selected = False
            return

        self.ensure_audio_agent()
            
        print("开始使用GIM音频编辑代理构建音乐Program...")
        
        try:
            # 使用传入的进度回调或默认的进度回调
            callback = progress_callback or self.update_program_progress
            # 使用GIM音频编辑代理从对话历史构建和渲染Program
            self.gim_program_result = self.audio_agent.build_and_render_program(
                self.chat_history,
                progress_callback=callback
            )
            
            print("GIM Program构建完成！")
            print(f"分析结果: {self.gim_program_result['analysis']}")
            print(f"Program阶段数: {len(self.gim_program_result['program'])}")
            print(f"输出文件: {self.gim_program_result['output']['file']}")
            
            # 为了兼容现有UI系统，将合成的音频作为"selected_music_tracks"
            # 这样UI可以展示最终的Program混音
            output_file = self.gim_program_result['output']['file']
            if self.condition == "kimusic":
                prepare_session_metadata(self.session_data, n_tracks=max(4, NUM_MUSIC_TRACKS))
                waypoint_sequence = self.session_data.get("waypoint_sequence") or []
                waypoint_va = (
                    waypoint_sequence[-1]
                    if waypoint_sequence
                    else self.session_data.get("target_state_va") or self.session_data.get("current_state_va")
                )
                proxy_track = create_proxy_generated_track(
                    waypoint_va=waypoint_va,
                    session_id=self.session_data.get("session_id"),
                    track_index=0,
                    fallback_audio_path=output_file,
                    waypoint_sequence=waypoint_sequence
                )
                proxy_track["duration_seconds"] = self.gim_program_result['output']['total_seconds']
                self.render_kimusic_track_once(proxy_track, output_file)
                self.selected_music_tracks = [proxy_track]
            else:
                self.selected_music_tracks = [{
                    'filename': os.path.basename(output_file),
                    'title': 'GIM Program Mix - Complete Therapy Session',
                    'full_path': output_file,  # 保存完整路径
                    'duration_seconds': self.gim_program_result['output']['total_seconds']
                }]
            self.append_music_tracks_to_session_data()
            
            self.music_selected = True
            print(f"GIM Program音频合成完成 - 输出文件: {output_file}")
            
        except Exception as e:
            print(f"GIM Program构建失败: {e}")
            import traceback
            traceback.print_exc()
            self.music_selected = False
            return

            # 如果GIM代理失败，回退到原始的音乐选择方法
            print("回退到原始音乐选择方法...")
            system_prompt, user_prompt = self.get_music_recommendation_prompts()
            criteria = self.music_db.get_music_criteria_json(system_prompt, user_prompt, self.api_key)
            self.selected_music_tracks = self.music_db.retrieve_music_for_therapy(criteria, num_tracks=NUM_MUSIC_TRACKS)
            self.append_music_tracks_to_session_data()
            self.music_selected = True
            print(f"回退音乐选择完成 - {len(self.selected_music_tracks)} tracks selected")

    def process_user_message(self, user_message):
        """处理用户消息，更新记忆"""
        self.ensure_memory_manager()
        if self.last_assistant_message:
            self.memory.process_message(user_message, self.last_assistant_message, api_key=self.api_key)
        else:
            self.memory.process_message(user_message, api_key=self.api_key)
        
        # 更新聊天历史
        # self.chat_history.append({"role": "user", "content": user_message}) # 删除这行重复的历史记录添加
            
    def get_ai_response(self):
        """获取AI响应 - 不再需要生成状态标签，因为状态分类已独立完成"""
        # 创建对话消息
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt()
            }
        ]
        
        # 检查是否需要总结对话
        self.ensure_memory_manager()
        if len(self.chat_history) >= self.summarization_threshold:
            print("!!!!!Conversation needs summarization!!!!!!!!!********")
            summary = self.memory.summarize_conversation(self.chat_history, api_key=self.api_key)
            if summary:
                print(f"Conversation summarized: {len(self.chat_history)} messages processed")
                # 如果对话很长，可以选择截断聊天历史以节省API调用中的令牌
                if len(self.chat_history) > 30:  # 如果对话很长
                    # 保留前几条消息作为上下文和最近的消息
                    self.chat_history = self.chat_history[:5] + self.chat_history[-15:]
                    print(f"Chat history truncated to {len(self.chat_history)} messages")
        
        # 添加清洗后的聊天历史作为上下文
        for msg in self.get_clean_chat_history_for_model():
            messages.append(msg)

        print("!!!!!Cleaned messages before LLM call:")
        print(messages)
        print("########################################################")
        
        # 准备API请求
        conn, api_path = make_llm_connection()
        payload = json.dumps({
            "model": MODEL_NAME,
            "max_tokens": MAX_TOKENS_RESPONSE,
            "messages": messages
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 发送API请求
        conn.request("POST", api_path, payload, headers)
        response = conn.getresponse()
        try:
            response_data = json.loads(response.read().decode("utf-8"))
        except json.decoder.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            response_text = "很抱歉，AI返回内容解析失败，请稍后重试。"
            self.last_assistant_message = response_text
            return response_text
        except Exception as e:
            print(f"获取AI响应时发生未知错误: {e}")
            response_text = "很抱歉，AI服务暂时不可用，请稍后重试。"
            self.last_assistant_message = response_text
            return response_text
        
        print("--------------------------------")
        print(response_data)
        
        # 从API响应中提取响应文本
        if isinstance(response_data, dict):
            if 'choices' in response_data:
                response_text = response_data['choices'][0]['message']['content']
            elif 'content' in response_data:
                response_text = response_data['content'][0]['text']
            else:
                print("Unexpected response format:", response_data)
                response_text = "I apologize, but I encountered an issue processing the response. Could you please try again?"
        else:
            print("Unexpected response type:", type(response_data))
            response_text = "I apologize, but I encountered an issue processing the response. Could you please try again?"
            
        print("Response text:", response_text)
        
        # 暂时存储助手的响应以供记忆处理
        self.last_assistant_message = response_text
        
        # 暂时更新聊天历史 - 注意：extract_state_from_response可能会修改这个内容
        self.append_chat_message("assistant", response_text)
        
        return response_text

    def get_ai_response_stream(self):
        """获取AI流式响应"""
        # 创建对话消息
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt()
            }
        ]
        
        # 检查是否需要总结对话
        self.ensure_memory_manager()
        if len(self.chat_history) >= self.summarization_threshold:
            print("!!!!!Conversation needs summarization!!!!!!!!!********")
            summary = self.memory.summarize_conversation(self.chat_history, api_key=self.api_key)
            if summary:
                print(f"Conversation summarized: {len(self.chat_history)} messages processed")
                # 如果对话很长，可以选择截断聊天历史以节省API调用中的令牌
                if len(self.chat_history) > 30:  # 如果对话很长
                    # 保留前几条消息作为上下文和最近的消息
                    self.chat_history = self.chat_history[:5] + self.chat_history[-15:]
                    print(f"Chat history truncated to {len(self.chat_history)} messages")
        
        # 添加聊天历史作为上下文
        for msg in self.get_clean_chat_history_for_model():
            messages.append(msg)

        print("!!!!!Cleaned messages before LLM call:")
        print(messages)
        print("########################################################")
        
        # 准备API请求 - 使用流式输出
        payload = json.dumps({
            "model": MODEL_NAME,
            "max_tokens": MAX_TOKENS_RESPONSE,
            "messages": messages,
            "stream": True  # 启用流式输出
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Connection': 'close'
        }
        
        # 发送流式API请求
        try:
            conn, api_path = make_llm_connection(timeout=None)
            conn.request("POST", api_path, payload, headers)
            response = conn.getresponse()
            
            if response.status == 200:
                full_response = ""
                pending_visible_text = ""
                token_detected = False
                token_tail_length = max(0, len(SESSION_OVER_TOKEN) - 1)
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    if token_detected:
                                        visible_text = content.replace(SESSION_OVER_TOKEN, "")
                                        if visible_text:
                                            yield visible_text
                                        continue
                                    pending_visible_text += content
                                    if SESSION_OVER_TOKEN in pending_visible_text:
                                        visible_text, _, after_token_text = pending_visible_text.partition(SESSION_OVER_TOKEN)
                                        if visible_text:
                                            yield visible_text
                                        if after_token_text:
                                            yield after_token_text
                                        pending_visible_text = ""
                                        token_detected = True
                                        continue
                                    if len(pending_visible_text) > token_tail_length:
                                        visible_text = pending_visible_text[:-token_tail_length]
                                        pending_visible_text = pending_visible_text[-token_tail_length:]
                                        if visible_text:
                                            yield visible_text
                        except json.JSONDecodeError:
                            continue
                if not token_detected and pending_visible_text:
                    yield pending_visible_text
                
                # 保存完整响应
                cleaned_response = self.strip_session_over_token(full_response)
                self.last_assistant_message = cleaned_response
                if cleaned_response.strip():
                    self.append_chat_message("assistant", cleaned_response)
                should_end_from_token = token_detected and not self.response_still_invites_continuation(cleaned_response)
                should_end_from_cap = (
                    self.current_state == GIMState.POSTLUDE
                    and self.phase_user_turns.get(GIMState.POSTLUDE, 0) >= POSTLUDE_MAX_USER_TURNS
                )
                if should_end_from_token or should_end_from_cap:
                    self.mark_ready_for_post_session_assessment()
                
            else:
                error_text = f"API请求失败: {response.status}"
                self.last_assistant_message = error_text
                yield error_text
                
        except Exception as e:
            error_text = f"流式请求失败: {str(e)}"
            self.last_assistant_message = error_text
            yield error_text
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # def extract_state_from_response(self, response_text):
    #     """从响应中提取状态"""
    #     state_match = re.search(r'<STATE>(.*?)</STATE>', response_text)
    #     if state_match:
    #         new_state = state_match.group(1)
            
    #         # 从响应中删除状态标签
    #         clean_response = re.sub(r'<STATE>.*?</STATE>\s*', '', response_text).strip()
            
    #         # 如果进入音乐成像阶段且已选择音乐，添加音乐提示信息
    #         if new_state == GIMState.MUSIC_IMAGING and self.music_selected and self.selected_music_tracks:
    #             # 添加音乐提示到响应中 - 使用当前语言的提示
    #             # 确保导入最新的MUSIC_NOTE变量，它会随语言切换而变化
    #             from prompts import MUSIC_NOTE
                
    #             # # 如果有GIM Program结果，添加额外的Program信息
    #             # if self.gim_program_result:
    #             #     program_info = f"\n\n**您的GIM音乐治疗Program已准备就绪**\n"
    #             #     program_info += f"- 治疗目标: {self.gim_program_result['analysis'].get('therapeutic_goal', '情感探索')}\n"
    #             #     program_info += f"- 当前情绪: {self.gim_program_result['analysis'].get('current_emotion', '待探索')}\n"
    #             #     program_info += f"- Program阶段: {len(self.gim_program_result['program'])}个阶段\n"
    #             #     program_info += f"- 总时长: {self.gim_program_result['output']['total_seconds']}秒\n"
    #             #     clean_response += program_info
                
    #             clean_response += MUSIC_NOTE
                
    #             print("已添加音乐提示信息到响应中")
            
    #         # 更新响应文本
    #         self.last_assistant_message = clean_response
            
    #         # 更新聊天历史中的最后一条消息
    #         if self.chat_history and self.chat_history[-1]["role"] == "assistant":
    #             self.chat_history[-1]["content"] = clean_response
            
    #         return new_state
        
    #     # 如果没有找到状态标签，保持当前状态
    #     return self.current_state



# 确保输出目录存在
os.makedirs("output", exist_ok=True)

def get_music_tracks(session: GIMTherapySession):
    """Get the currently selected music tracks for display"""
    music_files = []
    music_titles = []
    
    # Always check for selected music tracks, regardless of state
    if session.selected_music_tracks:
        print(f"Found {len(session.selected_music_tracks)} selected tracks")
        
        for track in session.selected_music_tracks:
            if "filename" in track:
                filename = track["filename"]
                title = track.get("title", filename)
                
                # 首先检查是否有full_path（GIM Program合成的文件）
                if "full_path" in track and os.path.exists(track["full_path"]) and os.path.isfile(track["full_path"]):
                    print(f"Found GIM Program file: {track['full_path']}")
                    music_files.append(track["full_path"])
                    music_titles.append(title)
                    continue
                
                # 检查相对路径和绝对路径
                if os.path.isabs(filename) and os.path.exists(filename) and os.path.isfile(filename):
                    print(f"Found music file (absolute path): {filename}")
                    music_files.append(filename)
                    music_titles.append(title)
                else:
                    # Check if file exists in the correct music directory
                    file_path = os.path.join("..", "toy_dataset", "mp3", filename)
                    print(f"Looking for music file: {file_path}")
                    
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        print(f"Found music file: {file_path}")
                        # 直接使用文件路径，Gradio会处理文件传输
                        music_files.append(file_path)
                        music_titles.append(title)
                    else:
                        print(f"WARNING: Music file not found: {file_path}")
                        # Try fallback locations
                        alt_path = os.path.join("toy_dataset", filename)
                        if os.path.exists(alt_path) and os.path.isfile(alt_path):
                            print(f"Found music file in alternate location: {alt_path}")
                            music_files.append(alt_path)
                            music_titles.append(title)
    
    print(f"Returning {len(music_files)} music files and titles")
    return music_files, music_titles


def export_session_results(session: GIMTherapySession):
    """Persist the unified session results to disk."""
    session_results = session.build_session_results_export()
    export_path = save_session_json(session_results, output_dir="session_results")
    print(f"SESSION RESULTS SAVED: {export_path}")
    return export_path

# def get_gim_program_info():
#     """获取GIM Program的详细信息用于显示,(暂时不展示programinfo)"""
#     if not therapy_session.gim_program_result:
#         return ""
    
#     result = therapy_session.gim_program_result
#     info = "## GIM音乐治疗Program详情\n\n"
    
#     # 分析结果
#     analysis = result.get('analysis', {})
#     info += f"**当前情绪状态:** {analysis.get('current_emotion', '未知')}\n"
#     info += f"**核心主题:** {', '.join(analysis.get('key_themes', []))}\n"
#     info += f"**治疗目标:** {analysis.get('therapeutic_goal', '未设定')}\n\n"
    
#     # Program阶段
#     info += "**Program阶段设计:**\n"
#     for i, phase in enumerate(result.get('program', []), 1):
#         info += f"{i}. **{phase.get('phase', f'阶段{i}')}** ({phase.get('duration_seconds', 0)}秒)\n"
#         info += f"   - 目的: {phase.get('purpose', '未说明')}\n"
#         criteria = phase.get('search_criteria', {})
#         if criteria.get('mood_keywords'):
#             info += f"   - 情绪关键词: {', '.join(criteria.get('mood_keywords', []))}\n"
#         info += f"   - 节奏偏好: {criteria.get('tempo_preference', '未设定')}\n"
#         info += f"   - 动态偏好: {criteria.get('dynamics_preference', '未设定')}\n\n"
    
#     # 输出信息
#     output_info = result.get('output', {})
#     info += f"**合成结果:**\n"
#     info += f"- 输出文件: {os.path.basename(output_info.get('file', ''))}\n"
#     info += f"- 总时长: {output_info.get('total_seconds', 0)}秒\n"
    
#     return info

# Create the Gradio interface with state management
with gr.Blocks(
    title="GIM Therapy Session",
    theme=gr.themes.Soft(),
    js="""() => {
        if (window.__gimExitWarningInstalled) {
            return;
        }
        window.__gimExitWarningInstalled = true;
        window.__gimExperimentComplete = false;
        window.__gimMusicPlaying = false;
        const getSelectedLanguage = () => {
            const checked = Array.from(document.querySelectorAll('input[type="radio"]:checked'))
                .map((input) => input.value || input.getAttribute("aria-label") || input.parentElement?.innerText || "")
                .join(" ");
            return checked.includes("English") ? "en" : "zh";
        };
        const getExitWarning = () => (
            getSelectedLanguage() === "en"
                ? "The experiment is not complete yet.\\nAre you sure you want to leave?\\nUnsaved data may be lost."
                : "实验尚未完成，确定要退出吗？\\n未完成的数据可能无法保存。"
        );
        const setChatLocked = (locked) => {
            const chatInput = document.querySelector("#gim-chat-input textarea");
            const submitButton = document.querySelector("#gim-submit-btn button");
            [chatInput, submitButton].forEach((element) => {
                if (!element) {
                    return;
                }
                element.disabled = locked;
                element.style.pointerEvents = locked ? "none" : "";
                element.style.opacity = locked ? "0.45" : "";
            });
        };
        const wireAudio = () => {
            document.querySelectorAll("audio").forEach((audio) => {
                if (audio.dataset.gimPlaybackWired) {
                    return;
                }
                audio.dataset.gimPlaybackWired = "1";
                audio.addEventListener("play", () => {
                    window.__gimMusicPlaying = true;
                    setChatLocked(true);
                });
                audio.addEventListener("ended", () => {
                    window.__gimMusicPlaying = false;
                    setChatLocked(false);
                });
            });
        };
        const observer = new MutationObserver(() => wireAudio());
        observer.observe(document.body, { childList: true, subtree: true });
        wireAudio();
        window.addEventListener("beforeunload", (event) => {
            if (document.querySelector(".final-completion-card")) {
                window.__gimExperimentComplete = true;
            }
            if (window.__gimExperimentComplete) {
                return;
            }
            const warning = getExitWarning();
            event.preventDefault();
            event.returnValue = warning;
            return warning;
        });
    }""",
    css="""
        .left-drawer {
            transition: all 0.3s ease-in-out;
        }
        .right-drawer {
            transition: all 0.3s ease-in-out;
        }
        .main-content {
            transition: all 0.3s ease-in-out;
        }
        .drawer-content {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 10px 0;
        }
        .progress-indicator {
            background: linear-gradient(90deg, #007bff, #0056b3);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .memory-item {
            background: #e9ecef;
            padding: 8px;
            border-radius: 5px;
            margin: 5px 0;
            border-left: 4px solid #007bff;
        }
        .program-phase {
            background: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            margin: 8px 0;
            border-left: 4px solid #28a745;
        }
        .final-completion-card {
            padding: 24px;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            background: #ffffff;
            color: #1f2328;
            line-height: 1.55;
        }
        .final-completion-card h2,
        .final-completion-card p {
            color: #1f2328;
        }
        .gradio-container .loading,
        .gradio-container .generating,
        .gradio-container .wrap.default,
        .gradio-container .wrap.pending {
            color: #1f2328 !important;
        }
        @media (max-width: 640px) {
            .final-completion-card {
                background: #ffffff;
                color: #111827;
                border-color: #c9d1d9;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }
            .gradio-container .loading,
            .gradio-container .generating,
            .gradio-container .wrap.default,
            .gradio-container .wrap.pending {
                background: rgba(255,255,255,0.96) !important;
                color: #111827 !important;
                border-radius: 8px;
            }
        }
    """
) as demo:
    # 会话状态管理
    session_state = gr.State(None)
    washout_timer = gr.Timer(1, active=False) if hasattr(gr, "Timer") else None
    
    # 添加语言切换
    current_language = gr.State(value="en")
    
    # 抽屉状态
    left_drawer_visible = gr.State(False)
    right_drawer_visible = gr.State(False)
    
    with gr.Column() as study_entry:
        with gr.Row():
            title_en = gr.Markdown("# Guided Imagery and Music (GIM) Therapy Session")
            title_zh = gr.Markdown("# 引导式音乐与意象 (GIM) 治疗会话", visible=False)
        
        with gr.Row():
            intro_en = gr.Markdown("""Welcome to your virtual GIM therapy session. This space is designed to guide you through a therapeutic journey 
        combining music and imagery. Feel free to share your thoughts and feelings openly.""")
            intro_zh = gr.Markdown("""欢迎您参加的虚拟GIM治疗会话。这个空间旨在引导您通过结合音乐和意象的治疗旅程。
            请随意开放地分享您的想法和感受。""", visible=False)

        study_instructions_en = gr.Markdown("""# Study Instructions

Welcome and thank you for participating in this study!

Please follow the steps below to complete the experiment:

1. Record the automatically generated **Participant ID**.
2. Complete the background questionnaire provided by the researcher (enter your Participant ID in the questionnaire).
3. Return to this page and click **Start Session**.
4. Follow the on-screen instructions to complete the pre-session questionnaires, therapeutic conversation, music experience, and post-music discussion.
5. After the chatbot completes the session, the system will **automatically** proceed to the post-session questionnaires.
6. After Session 1, please take a **5-minute break** as instructed by the system.
7. When the countdown finishes, click **Start Session 2** to begin the second session.
8. Session 2 follows the same procedure as Session 1.
9. After completing all questionnaires, the page will display **"Experiment Completed"**, indicating that the study has finished.

---

## Notes

- We recommend wearing headphones for the best experience.
- We recommend using a desktop or laptop computer with **Google Chrome** or **Microsoft Edge**.
- Please keep your internet connection stable. Do **not** refresh the page, close the browser window, or navigate away from the page during the experiment.
- Please keep your **Participant ID** in a safe place, as it is required for the background questionnaire.
- Music generation or loading may take several seconds. Please be patient. If necessary, you may click **Program** to check the generation progress.
- After the music finishes, the system will continue with a brief conversation. Please continue following the chatbot's guidance.
- When the system automatically proceeds to the post-session questionnaires, simply complete them as instructed. **There is no need to click "End Conversation."**
- During the experiment, you will mainly use the following buttons:
  - **Start Session**
  - **Submit Questionnaire**
  - **Send**
  - **Play Music**
  - **Start Session 2**
- There are no right or wrong answers. Please respond according to your genuine thoughts and feelings.
""")
        study_instructions_zh = gr.Markdown("""# 实验说明

欢迎参加本研究！

请按照以下步骤完成实验：

1. 记录系统自动生成的 Participant ID。
2. 完成研究人员提供的背景问卷（填写 Participant ID）。
3. 返回本页面，点击【开始会话】。
4. 根据页面提示完成会前量表、治疗对话、音乐体验以及音乐结束后的交流环节。
5. 当聊天机器人完成本次会话后，系统将自动进入会后量表。
6. 第一轮结束后，请按照系统提示休息 5 分钟。
7. 倒计时结束后，点击【开始第二次会话】完成第二轮体验。
8. 第二轮流程与第一轮相同。
9. 完成所有量表后，页面显示“实验完成”即表示实验结束。

注意事项

- 建议佩戴耳机完成实验。
- 建议使用电脑端（Chrome 或 Edge 浏览器）。
- 实验过程中请保持网络稳定，不要刷新页面、关闭浏览器或返回上一页。
- 请妥善保存 Participant ID，以便完成背景问卷。
- 音乐生成或加载可能需要几十秒，请耐心等待。如等待时间较长，可点击【程序】查看生成进度。
- 音乐播放结束后，系统仍会继续进行一段交流，请根据提示继续完成整个会话。
- 当系统自动进入会后量表时，请直接完成量表填写，无需手动点击【结束对话】。
- 除【开始会话】【提交量表】【发送】【音乐播放】【开始第二次会话】外，其余按钮一般无需使用。
- 所有问题均无标准答案，请根据自己的真实感受作答。
""", visible=False)
        
        # 语言切换开关
        with gr.Row():
            language_radio = gr.Radio(
                ["English", "中文"], 
                label="Language / 语言", 
                value="English",
                interactive=True
            )

        questionnaire_instruction_en = gr.Markdown("Questionnaire: please follow the study questionnaire link provided by the researcher, then click Start Experiment when you are ready.")
        questionnaire_instruction_zh = gr.Markdown("问卷：请按照研究者提供的问卷链接完成填写，准备好后点击开始实验。", visible=False)

        with gr.Row(elem_id="participant-controls") as participant_controls:
            initial_texts = get_ui_texts(False)
            user_id_input = gr.Textbox(
                label=initial_texts["participant_id_label"],
                placeholder=initial_texts["participant_id_placeholder"],
                value="",
                show_copy_button=True,
                interactive=True
            )
            start_session_btn = gr.Button(initial_texts["start_session"], variant="secondary")
        participant_status = gr.Markdown("")
    washout_display = gr.HTML("", visible=False)
    final_completion_display = gr.HTML("", visible=False)
    
    # 创建三栏抽屉式布局
    with gr.Row(visible=False) as therapy_workspace:
        # 左侧抽屉 - 用户记忆
        with gr.Column(scale=1, visible=False, elem_classes="left-drawer") as left_drawer:
            with gr.Column(elem_classes="drawer-content"):
                gr.Markdown("### 👤 User Memory / 用户记忆")
                user_memory_display = gr.Markdown("Memory content will be displayed here.\n用户记忆内容将在此显示。")
        
        # 中间主区域
        with gr.Column(scale=3, elem_classes="main-content") as main_content:
            # 创建聊天界面
            chatbot = gr.Chatbot(
                height=600,
                show_label=False,
                container=True,
                type="messages",
                value=[]
            )
            audio_player = gr.Audio(
                label=initial_texts["audio_label"],
                type="filepath",
                visible=False,
                interactive=False
            )

            with gr.Column(visible=False) as sam_panel:
                sam_title = gr.Markdown("### SAM Rating")
                sam_instruction = gr.Markdown("")
                sam_valence = gr.Slider(minimum=1, maximum=9, step=1, value=5, label="Valence", info="How pleasant or unpleasant do you feel right now?")
                sam_arousal = gr.Slider(minimum=1, maximum=9, step=1, value=5, label="Arousal", info="How calm or excited do you feel right now?")
                sam_submit_btn = gr.Button("Submit SAM", variant="secondary")
                sam_status = gr.Markdown("", visible=False)

            with gr.Column(visible=False) as panas_panel:
                panas_title = gr.Markdown("### PANAS Rating")
                panas_item_components = []
                with gr.Row():
                    with gr.Column():
                        for idx in range(10):
                            panas_item_components.append(
                                gr.Radio(
                                    choices=[1, 2, 3, 4, 5],
                                    value=3,
                                    label=f"PANAS Item {idx + 1}",
                                    interactive=True
                                )
                            )
                    with gr.Column():
                        for idx in range(10, 20):
                            panas_item_components.append(
                                gr.Radio(
                                    choices=[1, 2, 3, 4, 5],
                                    value=3,
                                    label=f"PANAS Item {idx + 1}",
                                    interactive=True
                                )
                            )
                panas_submit_btn = gr.Button("Submit PANAS", variant="secondary")
                panas_status = gr.Markdown("", visible=False)

            with gr.Column(visible=False) as sus_panel:
                sus_title = gr.Markdown("### System Usability Scale")
                sus_item_components = []
                for idx in range(10):
                    sus_item_components.append(
                        gr.Radio(
                            choices=[1, 2, 3, 4, 5],
                            value=3,
                            label=f"SUS Item {idx + 1}",
                            interactive=True
                        )
                    )
                sus_submit_btn = gr.Button("Submit SUS", variant="secondary")
                sus_status = gr.Markdown("", visible=False)

            with gr.Column(visible=False) as therapy_panel:
                therapy_title = gr.Markdown("### Therapy Experience")
                therapy_item_components = []
                for _, en_label, _ in THERAPY_EXPERIENCE_ITEMS:
                    therapy_item_components.append(
                        gr.Radio(
                            choices=[1, 2, 3, 4, 5, 6, 7],
                            value=4,
                            label=en_label,
                            interactive=True
                        )
                    )
                therapy_submit_btn = gr.Button("Submit Experience", variant="secondary")
                therapy_status = gr.Markdown("", visible=False)
             
            # 统一的消息输入框
            msg_input = gr.Textbox(
                label="Share your thoughts...",
                placeholder="Type your message here...",
                container=True,
                elem_id="gim-chat-input"
            )
            
            # 按钮行
            with gr.Row():
                toggle_memory_btn = gr.Button("👤 Memory", scale=0)
                submit_btn = gr.Button("Send", variant="primary", elem_id="gim-submit-btn")
                finish_session_btn = gr.Button("Finish Session", variant="secondary", interactive=False)
                toggle_program_btn = gr.Button("🎵 Program", scale=0)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation")
                save_btn = gr.Button("Save Session")
            
            # 保存状态文本
            save_info = gr.Textbox(label="Save Status", interactive=False)
        
        # 右侧抽屉 - Music Program信息
        with gr.Column(scale=1, visible=False, elem_classes="right-drawer") as right_drawer:
            with gr.Column(elem_classes="drawer-content"):
                program_title_en = gr.Markdown("### 🎵 Music Program", visible=True)
                program_title_zh = gr.Markdown("### 🎵 音乐程序", visible=False)
                
                # 进度显示区域
                progress_display = gr.Markdown("Waiting to generate music...\n等待生成音乐...", elem_classes="progress-indicator")
                
                # Program信息显示区域
                program_info_display = gr.Markdown("Program details will be displayed here.\n程序详情将在此显示。")
    
    def toggle_drawer(is_visible):
        """切换抽屉可见性的函数"""
        return gr.update(visible=not is_visible)

    def build_washout_display(user_id: str, washout: dict = None, session: GIMTherapySession = None):
        washout = washout or {}
        end_epoch = washout.get("washout_end_epoch")
        end_text = format_participant_timestamp(washout.get("washout_end") or "")
        if not end_epoch:
            return ""

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        remaining_seconds = max(0, int(float(end_epoch) - time.time()))
        remaining_text = format_remaining_time(remaining_seconds)
        return f"""
<div style="padding:12px;border:1px solid #ddd;border-radius:6px;background:#fafafa;">
  <p><strong>{texts["washout_title"]}</strong></p>
  <p>{texts["washout_id_intro"]}</p>
  <p style="font-size:1.4rem;font-weight:700;letter-spacing:0.04em;">{user_id}</p>
  <p>{texts["washout_instruction"]}</p>
  <p>{texts["washout_available_after"].format(end_time=end_text)}</p>
  <p style="font-size:1.6rem;font-weight:700;">{texts["washout_remaining"]} {remaining_text}</p>
  <p>{texts["washout_done"] if remaining_seconds <= 0 else texts["washout_continue"].format(minutes=remaining_seconds // 60, seconds=remaining_seconds % 60, participant_id=user_id)}</p>
</div>
"""

    def format_participant_timestamp(value):
        if not value:
            return ""
        return str(value).replace("T", " ")

    def format_remaining_time(seconds):
        seconds = max(0, int(seconds or 0))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def build_participant_save_message(user_id: str, session: GIMTherapySession = None):
        is_chinese = is_session_chinese(session)
        return get_ui_texts(is_chinese)["save_participant_id"].format(participant_id=user_id)

    def build_final_completion_display(session: GIMTherapySession = None):
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        return f"""
<div class="final-completion-card">
  <h2>{texts["final_complete_title"]}</h2>
  <p>{texts["final_complete_body"]}</p>
</div>
"""

    def is_final_completion(session: GIMTherapySession):
        if not session:
            return False
        total_sessions = len(session.condition_order or [session.condition])
        usability = session.session_data.get("usability", {})
        return (
            session.session_number >= total_sessions
            and usability.get("therapy_experience") is not None
        )

    def get_experiment_screen_updates(session: GIMTherapySession):
        if is_final_completion(session):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=build_final_completion_display(session), visible=True),
            )

        if getattr(session, "washout_pending", False):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            )

        if not getattr(session, "experiment_started", False):
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            )

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="", visible=False),
        )

    def get_washout_timer_update(active: bool):
        return (gr.update(active=bool(active)),) if washout_timer else ()

    def refresh_washout_screen(session: GIMTherapySession):
        if not session or not getattr(session, "washout_pending", False):
            return (
                gr.update(value="", visible=False),
                gr.update(),
                gr.update(),
                *get_washout_timer_update(False),
            )

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        end_epoch = session.session_data.get("washout_end_epoch")
        try:
            still_active = bool(end_epoch and float(end_epoch) > time.time())
        except (TypeError, ValueError):
            still_active = False
        if still_active:
            return (
                gr.update(value=build_washout_display(session.user_id, session.session_data, session=session), visible=True),
                gr.update(value=texts["start_session_2"]),
                gr.update(),
                *get_washout_timer_update(True),
            )

        return (
            gr.update(value="", visible=False),
            gr.update(value=texts["start_session_2"]),
            gr.update(visible=True),
            *get_washout_timer_update(False),
        )

    def build_participant_status_message(session: GIMTherapySession):
        if not session:
            return ""

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)

        if is_final_completion(session):
            return ""

        if getattr(session, "washout_pending", False):
            return ""

        if getattr(session, "session_completed", False):
            total_sessions = len(session.condition_order or [session.condition])
            if session.session_number >= total_sessions and session.panas_state.get("post_session_done"):
                return texts["all_sessions_complete"]
            return ""

        return texts["participant_id_generated"].format(session_number=session.session_number)

    def get_audio_player_update(session: GIMTherapySession):
        if not session or not session.selected_music_tracks:
            return gr.update(value=None, visible=False)

        audio_path = session.selected_music_tracks[0].get("full_path")
        if audio_path:
            audio_path = os.path.abspath(audio_path)
        if audio_path and os.path.exists(audio_path) and os.path.isfile(audio_path):
            print(f"Audio player path: {audio_path}")
            return gr.update(value=audio_path, visible=True)

        print(f"WARNING: Audio path not found for player: {audio_path}")
        return gr.update(value=None, visible=False)

    def get_chat_input_updates(session: GIMTherapySession):
        """Hide chat input controls until both pre-session assessments are complete."""
        if not session:
            session = GIMTherapySession()

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        ready = session.sam_state.get("pre_session_done") and session.panas_state.get("pre_session_done")
        if getattr(session, "session_completed", False):
            ready = False
        return (
            gr.update(
                visible=bool(ready),
                interactive=bool(ready),
                label=texts["input_label"],
                placeholder=texts["input_placeholder"]
            ),
            gr.update(visible=bool(ready), value=texts["submit"])
        )

    def get_finish_session_button_update(session: GIMTherapySession):
        return gr.update(interactive=bool(session and getattr(session, "session_completed", False)))

    def finish_session(session: GIMTherapySession):
        if not session:
            session = GIMTherapySession()

        if not getattr(session, "session_completed", False):
            return (
                session.chat_history,
                session,
                *get_sam_ui_updates(session),
                *get_panas_ui_updates(session),
                *get_chat_input_updates(session),
                *get_sus_ui_updates(session, reset_inputs=True),
                *get_therapy_ui_updates(session, reset_inputs=True)
            )

        session.mark_ready_for_post_session_assessment()

        return (
            session.chat_history,
            session,
            *get_sam_ui_updates(session),
            *get_panas_ui_updates(session),
            *get_chat_input_updates(session),
            *get_sus_ui_updates(session, reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True)
        )

    def get_sam_ui_updates(session: GIMTherapySession, status_message: str = "", reset_sliders: bool = False):
        """Build visibility and content updates for the minimal SAM panel."""
        if not session:
            session = GIMTherapySession()

        session.ensure_sam_pending_phase()
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        pending_phase = session.sam_state.get("pending_phase")

        if pending_phase == "pre_session" and not session.sam_state.get("pre_session_done"):
            title = texts["sam_title_pre"]
            visible = True
        elif pending_phase == "post_session" and not session.sam_state.get("post_session_done"):
            title = texts["sam_title_post"]
            visible = True
        else:
            title = texts["sam_title_default"]
            visible = False

        valence_update = (
            gr.update(label=texts["sam_valence_label"], info=texts["sam_valence_info"], value=5)
            if reset_sliders else
            gr.update(label=texts["sam_valence_label"], info=texts["sam_valence_info"])
        )
        arousal_update = (
            gr.update(label=texts["sam_arousal_label"], info=texts["sam_arousal_info"], value=5)
            if reset_sliders else
            gr.update(label=texts["sam_arousal_label"], info=texts["sam_arousal_info"])
        )
        status_update = gr.update(value=status_message, visible=bool(status_message))

        return (
            gr.update(visible=visible),
            gr.update(value=title),
            gr.update(value=texts["sam_instruction"], visible=visible),
            valence_update,
            arousal_update,
            status_update
        )

    def submit_sam_rating(session: GIMTherapySession, valence: int, arousal: int):
        """Store a SAM rating without affecting the existing chat/music flow."""
        if not session:
            session = GIMTherapySession()

        session.ensure_sam_pending_phase()
        pending_phase = session.sam_state.get("pending_phase")
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)

        if pending_phase not in ("pre_session", "post_session"):
            status_message = texts["sam_not_needed"]
            return (
                session,
                *get_sam_ui_updates(session, status_message=status_message),
                *get_panas_ui_updates(session)
            )

        rating_record = {
            "phase": pending_phase,
            "valence": int(valence),
            "arousal": int(arousal),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        session.sam_state["ratings"].append(rating_record)
        sam_payload = {
            "valence": rating_record["valence"],
            "arousal": rating_record["arousal"],
            "timestamp": rating_record["timestamp"]
        }
        if pending_phase == "pre_session":
            session.session_data["pre_session"]["sam"] = sam_payload
        elif pending_phase == "post_session":
            session.session_data["post_session"]["sam"] = sam_payload
        else:
            track_index = len(session.session_data.get("music_sequence", []))
            if track_index > 0:
                update_music_track_feedback(
                    session.session_data,
                    track_index,
                    rating_record["valence"],
                    rating_record["arousal"],
                    None,
                    None
                )

        if pending_phase == "pre_session":
            session.sam_state["pre_session_done"] = True
        elif pending_phase == "post_session":
            session.sam_state["post_session_done"] = True

        session.sam_state["pending_phase"] = None
        session.ensure_sam_pending_phase()
        session.ensure_panas_pending_phase()

        status_message = texts["sam_saved"]
        return (
            session,
            *get_sam_ui_updates(session, status_message=status_message, reset_sliders=True),
            *get_panas_ui_updates(session, reset_inputs=True)
        )

    def get_panas_ui_updates(session: GIMTherapySession, status_message: str = "", reset_inputs: bool = False):
        """Build visibility and content updates for the minimal PANAS panel."""
        if not session:
            session = GIMTherapySession()

        session.ensure_panas_pending_phase()
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        pending_phase = session.panas_state.get("pending_phase")

        if pending_phase == "pre_session" and not session.panas_state.get("pre_session_done"):
            title = texts["panas_title_pre"]
            visible = True
        elif pending_phase == "post_session" and not session.panas_state.get("post_session_done"):
            title = texts["panas_title_post"]
            visible = True
        else:
            title = texts["panas_title_default"]
            visible = False

        item_updates = []
        current_order = session.panas_state.get("current_order", [])
        for idx in range(20):
            item_key = current_order[idx] if idx < len(current_order) else None
            label = session.get_panas_item_label(item_key, is_chinese) if item_key else (f"条目 {idx + 1}" if is_chinese else f"PANAS Item {idx + 1}")
            update_kwargs = {"label": label}
            if reset_inputs:
                update_kwargs["value"] = 3
            item_updates.append(gr.update(**update_kwargs))

        status_update = gr.update(value=status_message, visible=bool(status_message))

        return (
            gr.update(visible=visible),
            gr.update(value=title),
            *item_updates,
            status_update
        )

    def get_sus_ui_updates(session: GIMTherapySession, status_message: str = "", reset_inputs: bool = False):
        """Build visibility and content updates for the SUS panel."""
        if not session:
            session = GIMTherapySession()

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        usability = session.session_data.get("usability", {})
        sus_done = usability.get("sus_responses") is not None
        visible = bool(session.panas_state.get("post_session_done")) and not sus_done and not getattr(session, "washout_pending", False)

        item_updates = []
        for idx, labels in enumerate(SUS_ITEMS):
            label = labels[1] if is_chinese else labels[0]
            update_kwargs = {"label": f"S{idx + 1}. {label}"}
            if reset_inputs:
                update_kwargs["value"] = 3
            item_updates.append(gr.update(**update_kwargs))

        return (
            gr.update(visible=visible),
            gr.update(value=texts["sus_title"] if visible else texts["sus_title_default"]),
            *item_updates,
            gr.update(value=status_message, visible=bool(status_message))
        )

    def calculate_sus_score(responses):
        score = 0
        for idx, response in enumerate(responses):
            if idx % 2 == 0:
                score += response - 1
            else:
                score += 5 - response
        return score * 2.5

    def submit_sus_rating(session: GIMTherapySession, *responses):
        """Store SUS responses and reveal the therapy experience form."""
        if not session:
            session = GIMTherapySession()

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        usability = session.session_data.setdefault("usability", {})
        if not session.panas_state.get("post_session_done"):
            return (
                session,
                *get_sus_ui_updates(session, status_message=texts["sus_not_needed"]),
                *get_therapy_ui_updates(session)
            )

        sus_responses = [int(response) if response is not None else 3 for response in responses[:10]]
        usability["sus_responses"] = sus_responses
        usability["sus_score"] = calculate_sus_score(sus_responses)

        return (
            session,
            *get_sus_ui_updates(session, status_message=texts["sus_saved"], reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True)
        )

    def get_therapy_ui_updates(session: GIMTherapySession, status_message: str = "", reset_inputs: bool = False):
        """Build visibility and content updates for the therapy experience panel."""
        if not session:
            session = GIMTherapySession()

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        usability = session.session_data.get("usability", {})
        sus_done = usability.get("sus_responses") is not None
        therapy_done = usability.get("therapy_experience") is not None
        visible = bool(sus_done) and not therapy_done and not getattr(session, "washout_pending", False)

        item_updates = []
        for _, en_label, zh_label in THERAPY_EXPERIENCE_ITEMS:
            update_kwargs = {"label": zh_label if is_chinese else en_label}
            if reset_inputs:
                update_kwargs["value"] = 4
            item_updates.append(gr.update(**update_kwargs))

        return (
            gr.update(visible=visible),
            gr.update(value=texts["therapy_title"] if visible else texts["therapy_title_default"]),
            *item_updates,
            gr.update(value=status_message, visible=bool(status_message))
        )

    def submit_therapy_experience_rating(session: GIMTherapySession, *ratings):
        """Store therapy experience responses and persist the completed session export."""
        if not session:
            session = GIMTherapySession()

        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)
        usability = session.session_data.setdefault("usability", {})
        if usability.get("sus_responses") is None:
            return (
                session,
                *get_therapy_ui_updates(session, status_message=texts["therapy_not_needed"]),
                *get_experiment_screen_updates(session),
                gr.update(value=build_participant_status_message(session)),
                gr.update(value="", visible=False),
                *get_washout_timer_update(False)
            )

        usability["therapy_experience"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": {
                item_key: int(ratings[idx]) if idx < len(ratings) and ratings[idx] is not None else 4
                for idx, (item_key, _, _) in enumerate(THERAPY_EXPERIENCE_ITEMS)
            }
        }
        export_session_results(session)

        return (
            session,
            *get_therapy_ui_updates(session, status_message=texts["therapy_saved"], reset_inputs=True),
            *get_experiment_screen_updates(session),
            gr.update(value=build_participant_status_message(session)),
            gr.update(value="", visible=False),
            *get_washout_timer_update(False)
        )

    def submit_panas_rating(session: GIMTherapySession, *ratings):
        """Store a PANAS rating without affecting existing chat/music/SAM flow."""
        if not session:
            session = GIMTherapySession()

        session.ensure_panas_pending_phase()
        pending_phase = session.panas_state.get("pending_phase")
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)

        if pending_phase not in ("pre_session", "post_session"):
            status_message = texts["panas_not_needed"]
            return (
                session,
                *get_panas_ui_updates(session, status_message=status_message),
                *get_chat_input_updates(session),
                *get_sus_ui_updates(session),
                *get_therapy_ui_updates(session),
                gr.update(value=""),
                gr.update(value="", visible=False),
                gr.update(value=session.user_id),
                *get_experiment_screen_updates(session),
                *get_washout_timer_update(False)
            )

        current_order = session.panas_state.get("current_order", [])
        items = {}
        for idx, item_key in enumerate(current_order):
            rating_value = ratings[idx] if idx < len(ratings) and ratings[idx] is not None else 3
            items[item_key] = int(rating_value)

        pa_score = sum(items.get(item_key, 0) for item_key in PANAS_POSITIVE_ITEMS)
        na_score = sum(items.get(item_key, 0) for item_key in PANAS_NEGATIVE_ITEMS)

        panas_record = {
            "phase": pending_phase,
            "items": items,
            "pa_score": pa_score,
            "na_score": na_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        session.panas_state["panas"].append(panas_record)
        if pending_phase == "pre_session":
            session.session_data["pre_session"]["panas"] = copy.deepcopy(panas_record)
        elif pending_phase == "post_session":
            session.session_data["post_session"]["panas"] = copy.deepcopy(panas_record)
        print("PANAS RECORD:", pending_phase, pa_score, na_score)

        if pending_phase == "pre_session":
            session.panas_state["pre_session_done"] = True
        elif pending_phase == "post_session":
            session.panas_state["post_session_done"] = True

        participant_message = ""
        countdown_html = ""
        if pending_phase == "post_session" and session.session_number == 1:
            washout = start_washout(session.user_id)
            session.washout_pending = True
            session.session_data.update({
                "washout_start": washout.get("washout_start"),
                "washout_end": washout.get("washout_end"),
                "washout_start_epoch": washout.get("washout_start_epoch"),
                "washout_end_epoch": washout.get("washout_end_epoch"),
            })
            participant_message = ""
            countdown_html = build_washout_display(session.user_id, washout, session=session)

        if pending_phase == "post_session":
            export_session_results(session)
            session.append_ablation_log_once()

        session.panas_state["pending_phase"] = None
        session.panas_state["current_order"] = []
        session.panas_state["current_order_phase"] = None
        session.ensure_panas_pending_phase()

        status_message = texts["panas_saved"]
        return (
            session,
            *get_panas_ui_updates(session, status_message=status_message, reset_inputs=True),
            *get_chat_input_updates(session),
            *get_sus_ui_updates(session, reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True),
            gr.update(value=participant_message),
            gr.update(value=countdown_html, visible=bool(countdown_html)),
            gr.update(value=session.user_id),
            *get_experiment_screen_updates(session),
            *get_washout_timer_update(bool(countdown_html))
        )

    def format_memory_for_display(session: GIMTherapySession):
        """格式化并返回用户记忆的Markdown文本"""
        if not session or not session.memory:
            return "No memory data yet.\n暂无记忆数据。"
        
        # 获取当前语言
        is_chinese = is_session_chinese(session)
        
        if is_chinese:
            memory_text = "#### 最近情绪状态：\n"
            emotions = [m.get('emotion', '未知') for m in session.memory.memories.get('emotional_states', [])[-3:]]
            if emotions:
                memory_text += "- " + "\n- ".join(emotions)
            else:
                memory_text += "暂无记录"
            
            memory_text += "\n\n#### 焦点意图：\n"
            focuses = [f.get('focus', '未知') for f in session.memory.memories.get('focus_intentions', [])[-2:]]
            if focuses:
                memory_text += "- " + "\n- ".join(focuses)
            else:
                memory_text += "暂无记录"
                
            memory_text += "\n\n#### 音乐偏好：\n"
            prefs = session.memory.memories.get('preferences', {}).get('music', [])
            if prefs:
                for pref in prefs[-3:]:
                    sentiment = "喜欢" if pref.get('sentiment') == 'like' else "不喜欢"
                    memory_text += f"- {sentiment} {pref.get('genre', '未知风格')}\n"
            else:
                memory_text += "暂无记录"
        else:
            memory_text = "#### Recent Emotions:\n"
            emotions = [m.get('emotion', 'unknown') for m in session.memory.memories.get('emotional_states', [])[-3:]]
            if emotions:
                memory_text += "- " + "\n- ".join(emotions)
            else:
                memory_text += "No records yet"
            
            memory_text += "\n\n#### Focus Intentions:\n"
            focuses = [f.get('focus', 'unknown') for f in session.memory.memories.get('focus_intentions', [])[-2:]]
            if focuses:
                memory_text += "- " + "\n- ".join(focuses)
            else:
                memory_text += "No records yet"
                
            memory_text += "\n\n#### Music Preferences:\n"
            prefs = session.memory.memories.get('preferences', {}).get('music', [])
            if prefs:
                for pref in prefs[-3:]:
                    sentiment = "Likes" if pref.get('sentiment') == 'like' else "Dislikes"
                    memory_text += f"- {sentiment} {pref.get('genre', 'unknown genre')}\n"
            else:
                memory_text += "No records yet"
        
        return memory_text

    def format_program_for_display(session: GIMTherapySession):
        """格式化并返回Music Program信息的Markdown文本"""
        if not session or not session.gim_program_result:
            is_chinese = is_session_chinese(session)
            return "No program generated yet.\n暂未生成程序。" if is_chinese else "No program generated yet."
        
        result = session.gim_program_result
        is_chinese = is_session_chinese(session)
        
        if is_chinese:
            program_text = f"#### 治疗目标: {result['analysis'].get('therapeutic_goal', '情感探索')}\n\n"
            program_text += f"#### 当前情绪: {result['analysis'].get('current_emotion', '待探索')}\n\n"
            program_text += "#### Program阶段:\n"
            for i, phase in enumerate(result.get('program', []), 1):
                program_text += f"{i}. **{phase.get('phase', f'阶段{i}')}** ({phase.get('duration_seconds', 0)}秒)\n"
                program_text += f"   - 目的: {phase.get('purpose', '未说明')}\n\n"
            
            if 'output' in result:
                program_text += f"#### 合成信息:\n"
                program_text += f"- 总时长: {result['output'].get('total_seconds', 0)}秒\n"
                program_text += f"- 文件: {os.path.basename(result['output'].get('file', ''))}\n"
        else:
            program_text = f"#### Therapeutic Goal: {result['analysis'].get('therapeutic_goal', 'Emotional exploration')}\n\n"
            program_text += f"#### Current Emotion: {result['analysis'].get('current_emotion', 'To be explored')}\n\n"
            program_text += "#### Program Phases:\n"
            for i, phase in enumerate(result.get('program', []), 1):
                program_text += f"{i}. **{phase.get('phase', f'Phase {i}')}** ({phase.get('duration_seconds', 0)}s)\n"
                program_text += f"   - Purpose: {phase.get('purpose', 'Not specified')}\n\n"
            
            if 'output' in result:
                program_text += f"#### Synthesis Info:\n"
                program_text += f"- Total duration: {result['output'].get('total_seconds', 0)}s\n"
                program_text += f"- File: {os.path.basename(result['output'].get('file', ''))}\n"
        
        return program_text
    

    def process_dialogue_stream(message: str, session: GIMTherapySession):
        """只处理对话流"""
        if not session:
            session = GIMTherapySession()
            
        session.append_chat_message("user", message)
        streamed_response = ""
        for chunk in session.get_next_response_stream(message):
            chunk_text = session._normalize_message_content(chunk)
            streamed_response += chunk_text
            yield (
                session.chat_history + [{"role": "assistant", "content": streamed_response}],
                session,
                get_finish_session_button_update(session)
            )
        yield session.chat_history, session, get_finish_session_button_update(session)

    def generate_music_stream(session: GIMTherapySession):
        """在需要时，流式生成音乐并更新UI"""
        print("DEBUG current_state =", session.current_state)
        print("DEBUG music_selected =", session.music_selected)
        session.ensure_sam_pending_phase()
        session.ensure_panas_pending_phase()
        is_chinese = is_session_chinese(session)
        texts = get_ui_texts(is_chinese)

        # 只有在 music_imaging 阶段且音乐未选择时才执行
        if session.current_state != GIMState.MUSIC_IMAGING or session.music_selected:
            yield (
                session.chat_history,
                session,
                format_memory_for_display(session),
                format_program_for_display(session),
                "Music generation not required.\n无需生成音乐。",
                get_audio_player_update(session),
                *get_sam_ui_updates(session),
                *get_panas_ui_updates(session),
                *get_chat_input_updates(session),
                get_finish_session_button_update(session)
            )
            return

        # 定义回调函数，用于在音乐生成时更新session状态
        def progress_callback(stage, status, progress, data=None):
            is_chinese = is_session_chinese(session)
            callback_texts = get_ui_texts(is_chinese)
            
            stage_names = {
                "analysis": "分析" if is_chinese else "Analysis",
                "design": "设计" if is_chinese else "Design", 
                "music_search": "音乐检索" if is_chinese else "Music Search",
                "processing": "音乐处理" if is_chinese else "Music Processing",
                "synthesis": "音频合成" if is_chinese else "Audio Synthesis"
            }
            
            session.progress_status = f"**{stage_names.get(stage, stage)}**: {status} ({progress:.0f}%)"

            if stage == "processing":
                session.append_guidance_message_once("music_processing_started", callback_texts["music_processing"])
            
            if data and stage == "synthesis" and progress == 100:
                session.gim_program_result = data
        
        # 使用线程运行耗时任务
        import threading
        session.append_guidance_message_once("music_generation_started", texts["music_start"])
        music_generation_thread = threading.Thread(
            target=session.select_music_for_imaging,
            kwargs={'progress_callback': progress_callback}
        )
        music_generation_thread.start()

        # 主线程循环检查进度并yield更新
        while music_generation_thread.is_alive():
            time.sleep(0.5)  # 每0.5秒更新一次UI
            yield (
                session.chat_history, 
                session, 
                format_memory_for_display(session), 
                format_program_for_display(session),
                session.progress_status,
                get_audio_player_update(session),
                *get_sam_ui_updates(session),
                *get_panas_ui_updates(session),
                *get_chat_input_updates(session),
                get_finish_session_button_update(session)
            )
        
        # 任务完成
        if session.selected_music_tracks or session.gim_program_result:
            session.music_selected = True
        else:
            session.music_selected = False
        
        # 将音乐播放器添加到对话历史
        if session.selected_music_tracks:
            audio_path = session.selected_music_tracks[0].get('full_path')
            if audio_path:
                audio_path = os.path.abspath(audio_path)
            if audio_path and os.path.exists(audio_path):
                session.append_guidance_message_once("music_ready", texts["music_ready"])
                session.append_guidance_message_once("music_experience", texts["music_experience"])
                # 添加音频文件作为单独的消息
        #        session.chat_history.append({"role": "assistant", "content": (audio_path,)})
                session.append_chat_message(
                    "assistant",
                    texts["play_music"],
                    phase=GIMState.MUSIC_IMAGING
                )

        # 最终的UI更新
        is_chinese = is_session_chinese(session)
        completed_message = "**完成!** 您的音乐已准备就绪。" if is_chinese else "**Completed!** Your music is ready."
        yield (
            session.chat_history, 
            session, 
            format_memory_for_display(session),
            format_program_for_display(session),
            completed_message,
            get_audio_player_update(session),
            *get_sam_ui_updates(session),
            *get_panas_ui_updates(session),
            *get_chat_input_updates(session),
            get_finish_session_button_update(session)
        )

    # 初始化函数
    def initialize_session():
        init_started_at = time.time()
        print("[startup] initialize_session start")
        participant_id = generate_participant_id()
        info = get_next_session_info(participant_id)
        session_id = init_ablation_session(info["user_id"], info["condition"])
        session = GIMTherapySession(
            startup_light=True,
            user_id=info["user_id"],
            condition=info["condition"],
            condition_order=info["condition_order"],
            session_number=info["session_number"],
            session_id=session_id,
        )
        print(f"[startup] initialize_session completed in {time.time() - init_started_at:.3f}s")
        return (
            session,
            gr.update(value=session.user_id),
            gr.update(value=build_participant_status_message(session)),
            gr.update(value="", visible=False),
            *get_experiment_screen_updates(session),
            *get_washout_timer_update(False),
            session.chat_history,
            format_memory_for_display(session),
            format_program_for_display(session),
            gr.update(value=None, visible=False),
            *get_sam_ui_updates(session),
            *get_panas_ui_updates(session),
            *get_chat_input_updates(session),
            get_finish_session_button_update(session),
            *get_sus_ui_updates(session, reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True)
        )

    def start_participant_session(user_id: str, current_session: GIMTherapySession = None):
        info = get_next_session_info(user_id)
        washout = info.get("washout") or {}
        language = getattr(current_session, "language", "en")
        is_chinese = language == "zh"
        texts = get_ui_texts(is_chinese)
        if info.get("washout_required"):
            session = GIMTherapySession(
                startup_light=True,
                user_id=info["user_id"],
                condition=info["condition"],
                condition_order=info["condition_order"],
                session_number=info["session_number"],
                language=language,
            )
            session.experiment_started = True
            session.session_completed = True
            session.washout_pending = True
            session.sam_state["pending_phase"] = None
            session.sam_state["pre_session_done"] = True
            session.sam_state["post_session_done"] = True
            session.panas_state["pre_session_done"] = True
            session.panas_state["post_session_done"] = True
            remaining = washout.get("remaining_seconds", 0)
            status = ""
            return (
                session,
                gr.update(value=info["user_id"]),
                session.chat_history,
                format_memory_for_display(session),
                format_program_for_display(session),
                gr.update(value=status),
                gr.update(value=build_washout_display(info["user_id"], washout, session=session), visible=True),
                gr.update(value=None, visible=False),
                *get_experiment_screen_updates(session),
                *get_washout_timer_update(True),
                *get_sam_ui_updates(session, reset_sliders=True),
                *get_panas_ui_updates(session, reset_inputs=True),
                *get_chat_input_updates(session),
                get_finish_session_button_update(session),
                *get_sus_ui_updates(session, reset_inputs=True),
                *get_therapy_ui_updates(session, reset_inputs=True)
            )

        session_id = init_ablation_session(info["user_id"], info["condition"])
        session = GIMTherapySession(
            startup_light=True,
            user_id=info["user_id"],
            condition=info["condition"],
            condition_order=info["condition_order"],
            session_number=info["session_number"],
            session_id=session_id,
            language=language,
        )
        session.experiment_started = True
        if info["all_sessions_complete"]:
            session.session_completed = True
            session.sam_state["pending_phase"] = None
            session.sam_state["pre_session_done"] = True
            session.sam_state["post_session_done"] = True
            session.panas_state["pre_session_done"] = True
            session.panas_state["post_session_done"] = True
            status = texts["all_sessions_complete"]
        else:
            status = texts["session_ready"].format(session_number=info["session_number"])

        return (
            session,
            gr.update(value=info["user_id"]),
            session.chat_history,
            format_memory_for_display(session),
            format_program_for_display(session),
            gr.update(value=status),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
            *get_experiment_screen_updates(session),
            *get_washout_timer_update(False),
            *get_sam_ui_updates(session, reset_sliders=True),
            *get_panas_ui_updates(session, reset_inputs=True),
            *get_chat_input_updates(session),
            get_finish_session_button_update(session),
            *get_sus_ui_updates(session, reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True)
        )
    
    # 清除对话函数
    def clear_conversation(session: GIMTherapySession):
        """清除对话但保留记忆"""
        if not session:
            session = GIMTherapySession()
            
        # 重置当前状态为PRELUDE
        session.current_state = GIMState.PRELUDE
        session.timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.music_selected = False
        session.selected_music_tracks = []
        session.gim_program_result = None
        session.kimusic_render_result = None
        session.baseline_used_track_ids = set()
        session.ablation_logged = False
        session.washout_pending = False
        
        # 清除进度显示相关属性
        session.progress_status = ""
        session.phase_info = ""
        session.music_info = ""
        
        # 重新初始化欢迎消息
        session.chat_history = []
        session.session_data = create_empty_session(user_id=session.user_id, condition=session.condition)
        session.session_data["condition_order"] = list(session.condition_order or [])
        session.session_data["session_number"] = session.session_number
        session.session_data["timestamp_start"] = session.timestamp_start
        session.initialize_welcome_message()
        session.sam_state = {
            "ratings": [],
            "pending_phase": "pre_session",
            "pre_session_done": False,
            "post_session_done": False
        }
        session.session_completed = False
        session.panas_state = {
            "panas": [],
            "pending_phase": None,
            "pre_session_done": False,
            "post_session_done": False,
            "current_order": [],
            "current_order_phase": None
        }
        session.ui_message_flags = {
            "music_generation_started": False,
            "music_processing_started": False,
            "music_ready": False,
            "music_experience": False,
            "music_ended": False,
            "post_music_reflection": False,
            "closure": False,
            "final_assessment_intro": False
        }
        session.postlude_reflection_prompt_index = None
        
        return (
            session.chat_history,
            session,
            format_memory_for_display(session),
            format_program_for_display(session),
            gr.update(value=None, visible=False),
            *get_experiment_screen_updates(session),
            *get_washout_timer_update(False),
            *get_sam_ui_updates(session, reset_sliders=True),
            *get_panas_ui_updates(session, reset_inputs=True),
            *get_chat_input_updates(session),
            get_finish_session_button_update(session),
            *get_sus_ui_updates(session, reset_inputs=True),
            *get_therapy_ui_updates(session, reset_inputs=True)
        )
    
    # 保存会话函数
    def save_session(session: GIMTherapySession):
        if not session:
            is_chinese = is_session_chinese(session)
            return "错误：会话未初始化" if is_chinese else "Error: Session not initialized"
        if session.memory:
            memories_dir = os.path.join("data", "memories")
            os.makedirs(memories_dir, exist_ok=True)
            session_id = session.session_data.get("session_id", "session")
            memory_filename = f"{session.user_id}_{session_id}.json"
            session.memory.save_memories_to_file(os.path.join(memories_dir, memory_filename))
        is_chinese = is_session_chinese(session)
        return "会话已成功保存！" if is_chinese else "Session saved successfully!"
    
    # 语言切换处理函数
    def change_language(lang_choice: str, session: GIMTherapySession):
        """切换界面语言并更新提示模板"""
        if not session:
            session = GIMTherapySession()

        # 保存真实对话历史（除了欢迎消息）
        true_chat_history = session.chat_history[1:] if len(session.chat_history) > 1 else []

        # 设置语言
        is_chinese = lang_choice != "English"
        session.language = "zh" if is_chinese else "en"
        
        # 重新初始化欢迎消息
        session.chat_history = []
        session.initialize_welcome_message()
        
        # 恢复真实对话历史
        session.chat_history.extend(true_chat_history)
        session.ensure_sam_pending_phase()
        session.ensure_panas_pending_phase()
        
        # 准备UI文本
        ui_text = {
            "clear": "清除对话" if is_chinese else "Clear Conversation",
            "save": "保存会话" if is_chinese else "Save Session",
            "memory": "👤 记忆" if is_chinese else "👤 Memory",
            "program": "🎵 程序" if is_chinese else "🎵 Program",
            "save_status": "保存状态" if is_chinese else "Save Status"
        }
        new_texts = get_ui_texts(is_chinese)
        chat_input_update, submit_update = get_chat_input_updates(session)
        washout_display_value = ""
        if getattr(session, "washout_pending", False):
            washout_display_value = build_washout_display(session.user_id, session.session_data, session=session)
        
        # 返回更新
        return (
            session.chat_history, session,
            format_memory_for_display(session),
            format_program_for_display(session),
            gr.update(visible=not is_chinese),  # title_en
            gr.update(visible=is_chinese),      # title_zh
            gr.update(visible=not is_chinese),  # intro_en
            gr.update(visible=is_chinese),      # intro_zh
            gr.update(visible=not is_chinese),  # study_instructions_en
            gr.update(visible=is_chinese),      # study_instructions_zh
            gr.update(visible=not is_chinese),  # questionnaire_instruction_en
            gr.update(visible=is_chinese),      # questionnaire_instruction_zh
            gr.update(visible=not is_chinese),  # program_title_en
            gr.update(visible=is_chinese),      # program_title_zh
            gr.update(label=new_texts["participant_id_label"], placeholder=new_texts["participant_id_placeholder"]),
            gr.update(value=new_texts["start_session"]),
            gr.update(label=new_texts["audio_label"]),
            chat_input_update,  # msg_input
            gr.update(value=ui_text["memory"]),    # toggle_memory_btn
            submit_update,    # submit_btn
            gr.update(value=new_texts["finish_session"], interactive=bool(getattr(session, "session_completed", False))),   # finish_session_btn
            gr.update(value=ui_text["program"]),   # toggle_program_btn
            gr.update(value=ui_text["clear"]),     # clear_btn
            gr.update(value=ui_text["save"]),      # save_btn
            gr.update(label=ui_text["save_status"]), # save_info
            *get_sam_ui_updates(session),
            *get_panas_ui_updates(session),
            gr.update(value=new_texts["sam_submit"]),   # sam_submit_btn
            gr.update(value=new_texts["panas_submit"]),  # panas_submit_btn
            *get_sus_ui_updates(session),
            gr.update(value=new_texts["sus_submit"]),   # sus_submit_btn
            *get_therapy_ui_updates(session),
            gr.update(value=new_texts["therapy_submit"]),  # therapy_submit_btn
            *get_experiment_screen_updates(session),
            gr.update(value=build_participant_status_message(session)),
            gr.update(value=washout_display_value, visible=bool(washout_display_value))
        )
    
    # 初始化应用
    demo.load(
        initialize_session,
        outputs=[
            session_state, user_id_input, participant_status, washout_display,
            study_entry, therapy_workspace, final_completion_display,
            *([washout_timer] if washout_timer else []),
            chatbot, user_memory_display, program_info_display, audio_player,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn, finish_session_btn,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status
        ]
    )

    start_session_btn.click(
        start_participant_session,
        [user_id_input, session_state],
        [
            session_state, user_id_input, chatbot, user_memory_display, program_info_display, participant_status, washout_display, audio_player,
            study_entry, therapy_workspace, final_completion_display,
            *([washout_timer] if washout_timer else []),
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn, finish_session_btn,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status
        ]
    )
    
    # 抽屉切换事件
    toggle_memory_btn.click(
        toggle_drawer,
        inputs=[left_drawer_visible],
        outputs=[left_drawer]
    ).then(
        lambda x: not x,
        inputs=[left_drawer_visible],
        outputs=[left_drawer_visible]
    )
    
    toggle_program_btn.click(
        toggle_drawer,
        inputs=[right_drawer_visible],
        outputs=[right_drawer]
    ).then(
        lambda x: not x,
        inputs=[right_drawer_visible],
        outputs=[right_drawer_visible]
    )
    
    # 提交按钮事件链
    submit_btn.click(
        process_dialogue_stream,
        [msg_input, session_state],
        [chatbot, session_state, finish_session_btn]
    ).then(
        lambda: "",
        None,
        [msg_input]  # 清空输入框
    ).then(
        generate_music_stream,
        [session_state],
        [
            chatbot, session_state, user_memory_display, program_info_display, progress_display, audio_player,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn,
            finish_session_btn
        ]
    )
    
    # 输入框回车提交
    msg_input.submit(
        process_dialogue_stream,
        [msg_input, session_state],
        [chatbot, session_state, finish_session_btn]
    ).then(
        lambda: "",
        None,
        [msg_input]  # 清空输入框
    ).then(
        generate_music_stream,
        [session_state],
        [
            chatbot, session_state, user_memory_display, program_info_display, progress_display, audio_player,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn,
            finish_session_btn
        ]
    )
    
    # 清除对话按钮
    clear_btn.click(
        clear_conversation,
        [session_state],
        [
            chatbot, session_state, user_memory_display, program_info_display, audio_player,
            study_entry, therapy_workspace, final_completion_display,
            *([washout_timer] if washout_timer else []),
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn, finish_session_btn,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status
        ]
    )
    
    # 保存会话按钮
    save_btn.click(
        save_session,
        [session_state],
        [save_info]
    )

    
    # 语言切换事件
    language_radio.change(
        change_language,
        inputs=[language_radio, session_state],
        outputs=[
            chatbot, session_state, user_memory_display, program_info_display,
            title_en, title_zh, intro_en, intro_zh,
            study_instructions_en, study_instructions_zh,
            questionnaire_instruction_en, questionnaire_instruction_zh,
            program_title_en, program_title_zh,
            user_id_input, start_session_btn, audio_player,
            msg_input, toggle_memory_btn, submit_btn, finish_session_btn, toggle_program_btn,
            clear_btn, save_btn, save_info,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            sam_submit_btn, panas_submit_btn,
            sus_panel, sus_title, *sus_item_components, sus_status, sus_submit_btn,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status, therapy_submit_btn,
            study_entry, therapy_workspace, final_completion_display,
            participant_status, washout_display
        ]
    )

    if washout_timer and hasattr(washout_timer, "tick"):
        washout_timer.tick(
            refresh_washout_screen,
            [session_state],
            [washout_display, start_session_btn, study_entry, washout_timer]
        )

    sam_submit_btn.click(
        submit_sam_rating,
        [session_state, sam_valence, sam_arousal],
        [
            session_state,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status
        ]
    )

    panas_submit_btn.click(
        submit_panas_rating,
        [session_state, *panas_item_components],
        [
            session_state,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status,
            participant_status, washout_display, user_id_input,
            study_entry, therapy_workspace, final_completion_display,
            *([washout_timer] if washout_timer else [])
        ]
    )

    finish_session_btn.click(
        finish_session,
        [session_state],
        [
            chatbot, session_state,
            sam_panel, sam_title, sam_instruction, sam_valence, sam_arousal, sam_status,
            panas_panel, panas_title, *panas_item_components, panas_status,
            msg_input, submit_btn,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status
        ]
    )

    sus_submit_btn.click(
        submit_sus_rating,
        [session_state, *sus_item_components],
        [
            session_state,
            sus_panel, sus_title, *sus_item_components, sus_status,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status
        ]
    )

    therapy_submit_btn.click(
        submit_therapy_experience_rating,
        [session_state, *therapy_item_components],
        [
            session_state,
            therapy_panel, therapy_title, *therapy_item_components, therapy_status,
            study_entry, therapy_workspace, final_completion_display,
            participant_status, washout_display,
            *([washout_timer] if washout_timer else [])
        ]
    )

if __name__ == "__main__":
    try:
        # Public demo deployment settings for Hugging Face Spaces.
        # DEMO_PASSWORD enables Gradio password protection.
        # DEMO_RATE_LIMIT_* controls lightweight IP-based rate limiting.
        os.makedirs("output/kimusic_generated", exist_ok=True)

        demo_username = os.getenv("DEMO_USERNAME", "reviewer")
        demo_password = os.getenv("DEMO_PASSWORD", "")
        demo_auth = (demo_username, demo_password) if demo_password else None

        if hasattr(demo, "queue"):
            demo.queue(
                default_concurrency_limit=int(os.getenv("DEMO_CONCURRENCY_LIMIT", "1")),
                max_size=int(os.getenv("DEMO_QUEUE_MAX_SIZE", "4")),
            )

        demo.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
            auth=demo_auth,
            auth_message="Kimusic EMNLP demo. Please use the reviewer password provided by the authors.",
            show_error=False,
            show_api=False,
            quiet=False,
            max_threads=int(os.getenv("DEMO_MAX_THREADS", "4")),
            allowed_paths=[
                os.path.abspath("output"),
                os.path.abspath("output/kimusic_generated"),
                os.path.abspath("demo_audio")
            ],
            app_kwargs={"middleware": make_security_middleware()},
        )
    except Exception as e:
        print(f"启动应用失败: {e}")
    finally:
        # 清理工作在这里不需要特定的therapy_session实例
        print("应用已关闭") 
