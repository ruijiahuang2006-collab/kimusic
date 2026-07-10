"""
GIM音频编辑代理系统
实现基于对话的智能音乐编辑和合成功能
"""
from __future__ import annotations


import os
import json
import re
import http.client
import time
import socket
import subprocess
import tempfile
import math
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import base64
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment
from pydub.effects import speedup
from pydub.utils import which

from hyper_parameters import MODEL_NAME,MUSIC_MODEL_NAME, MAX_TOKENS_RESPONSE, CANDIDATES_PER_PHASE


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


@dataclass
class ProgramPhase:
    phase: str
    duration_seconds: int
    purpose: str
    search_criteria: Dict[str, Any]


@dataclass
class MusicCandidate:
    """音乐候选项"""
    filename: str
    title: str
    duration_seconds: int
    metadata: Dict[str, Any] = None


@dataclass
class MusicSegment:
    """简单的音乐片段（向后兼容）"""
    filename: str
    duration_seconds: int
    title: str = None


@dataclass
class ProcessedMusicSegment:
    """处理后的音乐片段"""
    filename: str
    title: str
    start_time_seconds: float  # 从原音频的哪个时间点开始截取
    duration_seconds: float    # 截取的长度
    speed_ratio: float = 1.0   # 速度调整比例 (1.0 = 原速度)
    pitch_shift: float = 0.0   # 音调调整 (半音，0 = 不调整)
    volume_adjust: float = 1.0 # 音量调整 (1.0 = 原音量)
    fade_in_ms: int = 1000     # 淡入时长
    fade_out_ms: int = 1500    # 淡出时长
    processing_reason: str = ""  # 处理原因说明


class GIMAudioAgent:
    """GIM 音频编辑代理：
    1) 对话理解与意图分析
    2) Program 设计
    3) 音乐检索与分段
    4) 音频合成与编辑
    """

    def __init__(self, music_db, api_key: str, music_root: str = "../toy_dataset/mp3", output_dir: str = "output"):
        self.music_db = music_db
        self.api_key = api_key
        self.music_root = music_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 不再复用连接，避免远端关闭连接问题
        self.conn = None

    # ------------------------
    # 内部工具：ffmpeg 音频处理
    # ------------------------
    def _ffmpeg_tempo(self, audio: AudioSegment, tempo: float) -> AudioSegment:
        """使用ffmpeg的atempo滤镜调整速度（不改变音高）。支持0.5~2.0，超出范围拆分链式处理。"""
        if abs(tempo - 1.0) < 1e-3:
            return audio
        # 生成tempo因子链（每个在0.5~2.0范围内）
        def factors_for_tempo(t: float) -> List[float]:
            factors = []
            if t < 0.5:
                # 连续乘以0.5直到>=0.5
                while t < 0.5:
                    factors.append(0.5)
                    t /= 0.5
                if t != 1.0:
                    factors.append(t)
            elif t > 2.0:
                while t > 2.0:
                    factors.append(2.0)
                    t /= 2.0
                if t != 1.0:
                    factors.append(t)
            else:
                factors.append(t)
            return factors
        factors = factors_for_tempo(tempo)

        # 临时文件处理
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
            src_path = src.name
            audio.export(src_path, format="wav")
        current_input = src_path
        try:
            for i, f in enumerate(factors):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
                    dst_path = dst.name
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", current_input,
                    "-filter:a", f"atempo={f}",
                    dst_path
                ]
                subprocess.run(cmd, check=True)
                # 下次迭代以dst为输入
                if current_input != src_path and os.path.exists(current_input):
                    try:
                        os.unlink(current_input)
                    except Exception:
                        pass
                current_input = dst_path
            # 加载最终输出
            result = AudioSegment.from_file(current_input)
            return result
        finally:
            try:
                if os.path.exists(current_input):
                    os.unlink(current_input)
            except Exception:
                pass
            try:
                if os.path.exists(src_path):
                    os.unlink(src_path)
            except Exception:
                pass

    def _ffmpeg_pitch_shift(self, audio: AudioSegment, semitones: float) -> AudioSegment:
        """使用ffmpeg实现半音移调，同时尽量保持原有时长（使用asetrate+atempo校正）。"""
        if abs(semitones) < 1e-3:
            return audio
        factor = 2 ** (semitones / 12.0)
        sr = audio.frame_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
            src_path = src.name
            audio.export(src_path, format="wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
            dst_path = dst.name
        try:
            # 通过asetrate改变采样率从而改变音高，然后用atempo恢复节拍，最后aresample回原采样率
            # 注意：atempo范围为0.5~2.0，若factor超出范围，可以分段处理，但通常半音变化在此范围内
            # 若1/factor不在0.5~2.0，拆分链
            tempo_corr = 1.0 / factor
            if 0.5 <= tempo_corr <= 2.0:
                filter_str = f"asetrate={int(sr*factor)},atempo={tempo_corr},aresample={sr}"
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", src_path,
                    "-filter:a", filter_str,
                    dst_path
                ]
                subprocess.run(cmd, check=True)
            else:
                # 先改变采样率与重采样得到改变音高的音频（速度也变了），再用链式atempo拉回
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
                    tmp1_path = tmp1.name
                cmd1 = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", src_path,
                    "-filter:a", f"asetrate={int(sr*factor)},aresample={sr}",
                    tmp1_path
                ]
                subprocess.run(cmd1, check=True)
                # 链式tempo纠正
                corrected = self._ffmpeg_tempo(AudioSegment.from_file(tmp1_path), tempo_corr)
                corrected.export(dst_path, format="wav")
                try:
                    os.unlink(tmp1_path)
                except Exception:
                    pass
            result = AudioSegment.from_file(dst_path)
            return result
        finally:
            for p in [src_path, dst_path]:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # ------------------------
    # Step 1: 对话理解与治疗意图分析
    # ------------------------
    def analyze_dialogue(self, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        system_prompt = (
            "You are a clinical GIM therapist assistant. Read the conversation and extract a JSON with: "
            "current_emotion (one word or short phrase), key_themes (array), therapeutic_goal (string). "
            "Return ONLY JSON, no explanations."
        )

        # 拼接最相关的对话内容（控制长度）
        clipped_history = chat_history[-12:] if len(chat_history) > 12 else chat_history
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in clipped_history])

        user_prompt = (
            "Conversation transcripts (role: content):\n" + history_text +
            "\n\nExtract JSON strictly as: {\n  \"current_emotion\": string,\n  \"key_themes\": [string,...],\n  \"therapeutic_goal\": string\n}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = self._call_llm(MODEL_NAME,messages)
        data = self._extract_json(response_text)
        if not isinstance(data, dict):
            # 兜底
            data = {
                "current_emotion": "uncertain",
                "key_themes": ["exploration"],
                "therapeutic_goal": "provide safety and calm"
            }
        return data

    # ------------------------
    # Step 2: 设计 Program
    # ------------------------
    def design_program(self, analysis: Dict[str, Any]) -> List[ProgramPhase]:
        """使用LLM设计治疗Program，包含更智能的音乐搜索条件生成"""
        
        # 获取音乐库中可用的关键词选项（先设默认，避免 if 不进入 try 时未定义）
        default_mood = ["calm", "peaceful", "relaxing", "introspective", "contemplative", "uplifting", "energetic", "grounding", "flowing", "balanced"]
        default_genre = ["classical", "ambient", "instrumental", "piano", "orchestral", "electronic", "folk", "jazz", "meditation"]
        default_tag = ["nature", "meditation", "healing", "therapy", "mindfulness", "breathing", "centering", "flowing", "gentle", "soft"]
        default_theme = ["meditation", "therapy", "relaxation", "introspection", "healing", "mindfulness"]
        mood_options = default_mood.copy()
        genre_options = default_genre.copy()
        tag_options = default_tag.copy()
        theme_options = default_theme.copy()
        tempo_options = ["slow", "medium", "fast"]
        dynamics_options = ["soft", "moderate", "intense"]
        
        try:
            if self.music_db and hasattr(self.music_db, 'MusicDocumentManager') and self.music_db.MusicDocumentManager:
                mood_options = self.music_db.MusicDocumentManager.get_attribute_options("mood", 30)
                genre_options = self.music_db.MusicDocumentManager.get_attribute_options("genre", 30)
                # 获取标签选项
                tag_options = self.music_db.MusicDocumentManager.get_attribute_options("tags", 50)
                theme_options = self.music_db.MusicDocumentManager.get_attribute_options("theme", 30)
        except Exception as e:
            print(f"获取音乐库选项失败: {e}")
            # 使用默认选项（上面已初始化，此处可省略，保留以明确语义）
            mood_options = default_mood
            genre_options = default_genre
            tag_options = default_tag
            theme_options = default_theme
        
        # 构建系统提示词，包含可用的关键词选项
        system_prompt = (
            "You are an expert GIM (Guided Imagery and Music) therapist. "
            "Design a therapeutic program with multiple phases based on the client's needs. "
            "Each phase should have specific music selection criteria that can be matched against the available music library.\n\n"
            "Available music attributes to choose from:\n"
            f"- Moods: {', '.join(mood_options[:20])}\n"
            f"- Genres: {', '.join(genre_options[:20])}\n"
            f"- Tags: {', '.join(tag_options[:20])}\n"
            f"- Themes: {', '.join(theme_options[:20])}\n"
            f"- Tempo: {', '.join(tempo_options)}\n"
            f"- Dynamics: {', '.join(dynamics_options)}\n\n"
            "For each phase, provide search criteria that:\n"
            "1. Use keywords that exist in the available options above\n"
            "2. Are specific enough to find relevant music but not too restrictive\n"
            "3. Include both positive criteria (what to include) and negative criteria (what to avoid)\n"
            "4. Consider the therapeutic progression and emotional flow between phases\n\n"
            "Return a JSON array with each phase containing:\n"
            "- phase: descriptive name\n"
            "- duration_seconds: fixed short demo duration. Use exactly 20, 20, 25, 25 seconds for the four phases, summing to 90 seconds total.\n"
            "- purpose: therapeutic goal\n"
            "- search_criteria: music selection criteria with mood_keywords, genre_keywords, tempo_preference, dynamics_preference, avoid_keywords"
        )
        
        # 构建用户提示词
        user_prompt = (
            f"Client Analysis:\n"
            f"- Emotional State: {analysis.get('emotional_state', 'unknown')}\n"
            f"- Therapeutic Goals: {analysis.get('therapeutic_goals', 'general wellness')}\n"
            f"- Current Challenges: {analysis.get('current_challenges', 'not specified')}\n"
            f"- Preferred Duration: fixed 90 seconds total for demo/ablation testing\n\n"
            f"Design a GIM program with 3-4 phases that:\n"
            f"1. Addresses the client's emotional state and goals\n"
            f"2. Provides a natural therapeutic progression\n"
            f"3. Uses music that can actually be found in our library\n"
            f"4. Creates a balanced emotional journey\n\n"
            f"Remember to use only keywords that exist in the available options I provided above."
        )
        
        try:
            response = self._call_llm(MODEL_NAME,[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            phases_data = self._extract_json(response)
            if not phases_data or not isinstance(phases_data, list):
                print("LLM返回的格式无效，使用默认程序")
                print("!!response: ", response)
                return self._get_default_program(analysis)
            
            phases = []
            demo_durations = [20, 20, 25, 25]

            for phase_data in phases_data:
                try:
                    phase_idx = len(phases)

                    phase = ProgramPhase(
                        phase=phase_data.get("phase", "Unknown Phase"),
                        duration_seconds=demo_durations[phase_idx] if phase_idx < len(demo_durations) else 20,
                        purpose=phase_data.get("purpose", "General therapeutic support"),
                        search_criteria=phase_data.get("search_criteria", {})
                    )

                    phases.append(phase)

                except Exception as e:
                    print(f"解析阶段数据失败: {e}")
                    continue
            
            if not phases:
                print("没有成功解析任何阶段，使用默认程序")
                return self._get_default_program(analysis)
            
            print(f"设计了 {len(phases)} 个治疗阶段")
            for i, phase in enumerate(phases, 1):
                print(f"  {i}. {phase.phase} ({phase.duration_seconds}s): {phase.purpose}")
                print(f"     搜索条件: {phase.search_criteria}")
            
            return phases
            
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return self._get_default_program(analysis)
    
    def _get_default_program(self, analysis: Dict[str, Any]) -> List[ProgramPhase]:
        """当LLM失败时提供默认的治疗程序"""
        raise NotImplementedError("默认程序未实现")
        emotional_state = analysis.get('emotional_state', 'neutral')
        
        # 基于情绪状态提供默认程序
        if 'overwhelmed' in emotional_state.lower() or 'stressed' in emotional_state.lower():
            return [
                ProgramPhase("1. Grounding", 300, "Establish safety and presence", {
                    "mood_keywords": ["calm", "gentle", "soothing", "grounding"],
                    "tempo_preference": "slow",
                    "dynamics_preference": "soft",
                    "avoid_keywords": ["chaotic", "intense", "dramatic", "urgent"]
                }),
                ProgramPhase("2. Acknowledging Pressure", 300, "Recognize and validate current challenges", {
                    "mood_keywords": ["tension", "weight", "struggle", "building"],
                    "tempo_preference": "medium",
                    "dynamics_preference": "moderate",
                    "avoid_keywords": ["aggressive", "violent", "chaotic"]
                }),
                ProgramPhase("3. Finding Inner Space", 300, "Create internal spaciousness and breathing room", {
                    "mood_keywords": ["spacious", "expansive", "open", "breathing", "flowing"],
                    "tempo_preference": "medium",
                    "dynamics_preference": "moderate",
                    "avoid_keywords": ["constricted", "heavy", "frantic"]
                }),
                ProgramPhase("4. Rhythmic Balance", 300, "Establish healthy internal rhythms and flow", {
                    "mood_keywords": ["rhythmic", "balanced", "flowing", "structured", "playful"],
                    "tempo_preference": "medium",
                    "dynamics_preference": "moderate",
                    "avoid_keywords": ["rigid", "monotonous", "frenzied"]
                }),
                ProgramPhase("5. Integration", 300, "Bring together insights and new patterns", {
                    "mood_keywords": ["integrated", "whole", "harmonious", "peaceful", "complete"],
                    "tempo_preference": "slow",
                    "dynamics_preference": "soft",
                    "avoid_keywords": ["fragmented", "chaotic", "unresolved"]
                })
            ]
        else:
            # 通用程序
            return [
                ProgramPhase("1. Opening", 300, "Create safe therapeutic space", {
                    "mood_keywords": ["calm", "welcoming", "gentle", "peaceful"],
                    "tempo_preference": "slow",
                    "dynamics_preference": "soft",
                    "avoid_keywords": ["intense", "chaotic", "aggressive"]
                }),
                ProgramPhase("2. Exploration", 300, "Deepen into therapeutic focus", {
                    "mood_keywords": ["introspective", "contemplative", "thoughtful", "curious"],
                    "tempo_preference": "medium",
                    "dynamics_preference": "moderate",
                    "avoid_keywords": ["distracting", "overwhelming", "superficial"]
                }),
                ProgramPhase("3. Transformation", 300, "Support positive change and insight", {
                    "mood_keywords": ["flowing", "transforming", "evolving", "expanding"],
                    "tempo_preference": "medium",
                    "dynamics_preference": "moderate",
                    "avoid_keywords": ["stuck", "rigid", "confining"]
                }),
                ProgramPhase("4. Integration", 300, "Consolidate gains and prepare for closure", {
                    "mood_keywords": ["integrated", "whole", "harmonious", "complete"],
                    "tempo_preference": "slow",
                    "dynamics_preference": "soft",
                    "avoid_keywords": ["fragmented", "unresolved", "abrupt"]
                })
            ]

    # ------------------------
    # Step 3: 音乐检索与候选项收集
    # ------------------------
    def retrieve_music_candidates(self, phases: List[ProgramPhase], candidates_per_phase: int = 1) -> Dict[str, List[MusicCandidate]]:
        """为每个阶段检索多个音乐候选项，使用多级回退策略确保每个阶段都有候选项"""
        phase_candidates = {}
        
        for phase in phases:
            criteria = phase.search_criteria or {}
            candidates = []
            
            # 策略1: 使用原始条件进行精确搜索
            candidates = self._search_with_criteria(criteria, candidates_per_phase)
            
            # 策略2: 如果精确搜索失败，使用宽松条件搜索
            if len(candidates) < candidates_per_phase:
                print(f"阶段 '{phase.phase}' 精确搜索只找到 {len(candidates)} 个候选项，尝试宽松搜索...")
                relaxed_criteria = self._create_relaxed_criteria(criteria)
                additional_candidates = self._search_with_criteria(relaxed_criteria, candidates_per_phase - len(candidates))
                candidates.extend(additional_candidates)
            
            # 策略3: 如果仍然不够，使用通用条件搜索
            if len(candidates) < candidates_per_phase:
                print(f"阶段 '{phase.phase}' 宽松搜索后只有 {len(candidates)} 个候选项，使用通用搜索...")
                general_criteria = self._create_general_criteria(phase.purpose)
                additional_candidates = self._search_with_criteria(general_criteria, candidates_per_phase - len(candidates))
                candidates.extend(additional_candidates)
            
            # 策略4: 最后的回退 - 随机选择
            if len(candidates) < candidates_per_phase:
                print(f"阶段 '{phase.phase}' 搜索后只有 {len(candidates)} 个候选项，使用随机选择...")
                random_candidates = self._get_random_candidates(candidates_per_phase - len(candidates))
                candidates.extend(random_candidates)
            
            # 验证文件存在性并创建MusicCandidate对象
            valid_candidates = []
            for track in candidates:
                filename = track.get("filename", "")
                title = track.get("title", filename)
                
                # 如果只有文件名，拼到根目录
                if filename:
                    candidate_path = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "../toy_dataset/mp3", os.path.basename(filename))
                    )
                    if os.path.exists(candidate_path):
                        filename = candidate_path

                if filename and os.path.exists(filename):

                    # 获取音频文件的真实时长
                    try:
                        audio = AudioSegment.from_file(filename)
                        actual_duration = len(audio) / 1000.0  # 转换为秒
                    except Exception:
                        actual_duration = track.get("duration", 300)  # 默认5分钟
                    
                    valid_candidates.append(MusicCandidate(
                        filename=filename,
                        title=title,
                        duration_seconds=actual_duration,
                        metadata=track
                    ))
            
            phase_candidates[phase.phase] = valid_candidates
            print(f"阶段 '{phase.phase}' 最终找到 {len(valid_candidates)} 个有效候选项")
        
        return phase_candidates
    
    def _search_with_criteria(self, criteria: Dict[str, Any], num_tracks: int) -> List[Dict[str, Any]]:
        """使用给定条件搜索音乐"""
        try:
            tracks = self.music_db.retrieve_music_for_therapy(criteria, num_tracks=num_tracks) or []
            return tracks
        except Exception as e:
            print(f"搜索失败: {e}")
            for k in ["tempo_preference", "dynamics_preference"]:
                val = relaxed_criteria.get(k)
                if isinstance(val, list) and len(val) > 0:
                   relaxed_criteria[k] = val[0]
    
    def _create_relaxed_criteria(self, original_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """创建宽松的搜索条件"""
        relaxed_criteria = original_criteria.copy()
        #raise NotImplementedError("创建宽松的搜索条件未实现")
        for k in ["tempo_preference", "dynamics_preference"]:
            val = relaxed_criteria.get(k)
            if isinstance(val, list) and len(val) > 0:
                relaxed_criteria[k] = val[0]
        
        # 放宽tempo限制
        if "tempo_preference" in relaxed_criteria:
            tempo_mapping = {
                "slow": "medium",      # 慢速 -> 中速
                "medium": "medium",    # 中速保持不变
                "fast": "medium"       # 快速 -> 中速
            }
            relaxed_criteria["tempo_preference"] = tempo_mapping.get(
                relaxed_criteria["tempo_preference"], "medium"
            )
        
        # 放宽dynamics限制
        if "dynamics_preference" in relaxed_criteria:
            dynamics_mapping = {
                "soft": "moderate",     # 柔和 -> 中等
                "moderate": "moderate", # 中等保持不变
                "intense": "moderate"   # 强烈 -> 中等
            }
            relaxed_criteria["dynamics_preference"] = dynamics_mapping.get(
                relaxed_criteria["dynamics_preference"], "moderate"
            )
        
        # 减少mood关键词，只保留最核心的
        if "mood_keywords" in relaxed_criteria and relaxed_criteria["mood_keywords"]:
            # 保留前2个关键词
            relaxed_criteria["mood_keywords"] = relaxed_criteria["mood_keywords"][:2]
        
        # 减少避免关键词
        if "avoid_keywords" in relaxed_criteria and relaxed_criteria["avoid_keywords"]:
            # 只保留最关键的避免词
            relaxed_criteria["avoid_keywords"] = relaxed_criteria["avoid_keywords"][:1]
        
        return relaxed_criteria
    
    def _create_general_criteria(self, purpose: str) -> Dict[str, Any]:
        return {
            "mood_keywords": [
                "calm", "soft", "relaxing", "peaceful", "dreamy",
                "hopeful", "bright", "classical", "piano", "background"
            ],
            "genre_keywords": [
                "modern classical", "solo piano", "classical", "instrumental"
            ],
            "tempo_preference": "medium",
            "dynamics_preference": "moderate",
            "avoid_keywords": []
    }
    
    def _get_random_candidates(self, num_tracks: int) -> List[Dict[str, Any]]:
        """获取随机音乐候选项"""
        raise NotImplementedError("获取随机音乐候选项未实现")
        try:
            # 使用通用条件获取随机音乐
            general_criteria = {
                "mood_keywords": ["calm", "peaceful"],
                "tempo_preference": "medium",
                "dynamics_preference": "moderate",
                "avoid_keywords": []
            }
            
            # 尝试获取更多候选项，然后随机选择
            all_tracks = self.music_db.retrieve_music_for_therapy(general_criteria, num_tracks=20) or []
            
            # 随机选择指定数量的候选项
            import random
            if len(all_tracks) > num_tracks:
                return random.sample(all_tracks, num_tracks)
            else:
                return all_tracks
                
        except Exception as e:
            print(f"随机选择失败: {e}")
            # 返回空列表，让上层处理
            return []

    def _encode_base64_content_from_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


    def _get_audio_format(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            ".mp3": "mp3",
            ".wav": "wav",
            ".flac": "flac",
            ".m4a": "m4a",
            ".ogg": "ogg",
            ".aac": "aac",
        }
        return mapping.get(ext, "mp3")

    # ------------------------
    # Step 4: 智能音乐选择与处理设计
    # ------------------------
    def design_music_processing_parallel(self, phases: List[ProgramPhase],
                                      phase_candidates: Dict[str, List[MusicCandidate]],
                                      progress_callback: Optional[callable] = None) -> List[ProcessedMusicSegment]:
        """使用并行LLM调用为每个阶段智能选择音乐并设计处理方案
        
        这是design_music_processing的并行优化版本，通过并行处理多个阶段来减少总体延迟。
        如果并行处理失败，会回退到串行版本。
        """
        try:
            # 创建线程池
            with ThreadPoolExecutor(max_workers=min(4, len(phases))) as executor:
                # 准备所有阶段的处理任务
                future_to_phase = {}
                
                # 预加载所有需要的音频文件
                audio_cache = {}
                for phase_name, candidates in phase_candidates.items():
                    if candidates:
                        filepath = candidates[0].filename
                        if filepath and os.path.exists(filepath):
                            try:
                                audio_format = self._get_audio_format(filepath)
                                audio_b64 = self._encode_base64_content_from_file(filepath)
                                audio_cache[phase_name] = (audio_format, audio_b64)
                            except Exception as e:
                                print(f"预加载音频文件失败 {filepath}: {e}")
                
                # 汇总所有阶段简介用于全局规划
                phases_overview_lines = []
                for idx, ph in enumerate(phases, 1):
                    phases_overview_lines.append(
                        f"{idx}. {ph.phase} | duration={ph.duration_seconds}s | purpose={ph.purpose} | criteria={json.dumps(ph.search_criteria, ensure_ascii=False)}"
                    )
                phases_overview = "\n".join(phases_overview_lines)
                
                # 提交所有处理任务
                for phase in phases:
                    candidates = phase_candidates.get(phase.phase, [])
                    if not candidates:
                        continue
                        
                    # 准备任务参数
                    task_args = (
                        phase,
                        candidates,
                        phases_overview,
                        audio_cache.get(phase.phase)
                    )
                    
                    # 提交任务到线程池
                    future = executor.submit(
                        self._process_single_phase,
                        *task_args
                    )
                    future_to_phase[future] = phase
                
                # 收集所有结果
                processed_segments = []
                for future in concurrent.futures.as_completed(future_to_phase):
                    phase = future_to_phase[future]
                    try:
                        segment = future.result()
                        if segment:
                            processed_segments.append(segment)
                            if progress_callback:
                                progress_callback(phase.phase, "处理完成", True)
                            print(f"阶段 '{phase.phase}' 处理完成")
                    except Exception as e:
                        print(f"处理阶段 '{phase.phase}' 失败: {e}")
                        # 记录错误但继续处理其他阶段
                        continue
                
                return processed_segments
                
        except Exception as e:
            print(f"并行处理失败，回退到串行版本: {e}")
            # 回退到原始串行版本
            return self.design_music_processing(phases, phase_candidates)
    
    def _process_single_phase(self, phase: ProgramPhase, 
                            candidates: List[MusicCandidate],
                            phases_overview: str,
                            audio_data: Optional[Tuple[str, str]] = None) -> Optional[ProcessedMusicSegment]:
        """处理单个阶段的音乐设计（供并行处理使用）"""
        try:
            # 构建LLM提示词
            system_prompt = (
                "You are an expert in GIM therapy and audio processing. Given a therapy phase and music candidates, select the best segment and design appropriate audio processing parameters. Output ONLY a valid JSON that strictly matches this schema (no extra text, no comments, no trailing commas):\n\n"
                "{\n"
                "  \"start_time_seconds\": <float, seconds from beginning, >= 0>,\n"
                "  \"duration_seconds\": <float, seconds, > 0>,\n"
                "  \"speed_ratio\": <float, 1.0=normal, 0.8=20% slower, 1.2=20% faster>,\n"
                "  \"pitch_shift\": <int, semitones, 0=no change, +2=2 semitones higher, -1=1 semitone lower>,\n"
                "  \"volume_adjust\": <float, 1.0=normal, 1.5=50% louder, 0.7=30% quieter>,\n"
                "  \"fade_in_ms\": <int, milliseconds, typically 1000-5000, 0 if not needed>,\n"
                "  \"fade_out_ms\": <int, milliseconds, typically 1000-5000, 0 if not needed>,\n"
                "  \"processing_reason\": <string, concise rationale for choices>\n"
                "}\n\n"
                "Rules:\n"
                "- Return JSON only. Do not include explanations or any additional text outside the JSON.\n"
                "- Choose a musically coherent segment that supports therapeutic continuity and smooth transitions.\n"
                "- Avoid unnecessary edits (use speed_ratio=1.0, pitch_shift=0, volume_adjust=1.0 if already appropriate)."
            )
            
            # 准备候选音乐信息
            candidates_info = []
            for i, candidate in enumerate(candidates):
                temp_metadata = {}
                temp_metadata["filename"] = candidate.filename
                temp_metadata["tags"] = candidate.metadata.get("tags", [])
                temp_metadata["genre"] = candidate.metadata.get("genre", [])
                temp_metadata["mood"] = candidate.metadata.get("mood", [])
                temp_metadata["movement"] = candidate.metadata.get("movement", [])
                temp_metadata["theme"] = candidate.metadata.get("theme", [])
                temp_metadata["tempo"] = candidate.metadata.get("tempo", [])
                temp_metadata["dynamics_rmse_mean"] = candidate.metadata.get("dynamics_rmse_mean", [])

                info = {
                    "index": i,
                    "title": candidate.title,
                    "duration": candidate.duration_seconds,
                    "metadata": temp_metadata or {}
                }
                candidates_info.append(info)
            
            user_prompt = (
                "All Phases Overview (for better global planning and transitions):\n" +
                phases_overview +
                "\n\n" +
                f"Current Phase: {phase.phase}\n" +
                f"Required duration: {phase.duration_seconds} seconds\n" +
                f"Music candidate information:\n{json.dumps(candidates_info, ensure_ascii=False, indent=2)}\n\n" +
                "Instructions:\n"
                "- Select the most suitable segment from a candidate to support therapeutic continuity and smooth transitions.\n"
                "- Output ONLY the JSON defined in the system message. No extra text.\n"
                "- If the original segment already fits well, prefer keeping speed_ratio=1.0, pitch_shift=0, volume_adjust=1.0.\n"
            )

            # 准备消息
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # 如果有音频数据，添加到消息中
            if audio_data:
                audio_format, audio_b64 = audio_data
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                })
            else:
                messages.append({"role": "user", "content": user_prompt})
            
            # 调用LLM（音频模型需要更长的处理时间）
            response_text = self._call_llm(MUSIC_MODEL_NAME, messages, timeout=180)  # 3分钟超时
            if not response_text:
                print(f"音频处理阶段 {phase.phase} 超时或失败，使用默认参数")
                processing_design = None
            else:
                processing_design = self._extract_json(response_text)
            
            if not processing_design or not isinstance(processing_design, dict):
                # 兜底：选择第一个候选项，使用默认参数
                selected_candidate = candidates[0]
                processing_design = {
                    "start_time_seconds": 0.0,
                    "duration_seconds": min(phase.duration_seconds, selected_candidate.duration_seconds),
                    "speed_ratio": 1.0,
                    "pitch_shift": 0.0,
                    "volume_adjust": 1.0,
                    "fade_in_ms": 1000,
                    "fade_out_ms": 1500,
                    "processing_reason": "Default selection due to LLM processing failure"
                }
            
            # 验证并应用选择
            selected_candidate = candidates[0]  # 当前实现只使用第一个候选项
            
            # 确保参数合理性
            start_time = max(0, float(processing_design.get("start_time_seconds", 0)))
            duration = min(
                phase.duration_seconds,
                float(processing_design.get("duration_seconds", phase.duration_seconds)),
                selected_candidate.duration_seconds - start_time
            )
            
            # 参数范围验证和默认值设置
            speed_ratio = float(processing_design.get("speed_ratio", 1.0))
            pitch_shift = float(processing_design.get("pitch_shift", 0.0))
            volume_adjust = float(processing_design.get("volume_adjust", 1.0))
            fade_in_ms = int(processing_design.get("fade_in_ms", 1000))
            fade_out_ms = int(processing_design.get("fade_out_ms", 1500))
            
            processed_segment = ProcessedMusicSegment(
                filename=selected_candidate.filename,
                title=selected_candidate.title,
                start_time_seconds=start_time,
                duration_seconds=duration,
                speed_ratio=speed_ratio,  # 在音频处理时再进行范围限制
                pitch_shift=pitch_shift,   # 在音频处理时再进行范围限制
                volume_adjust=volume_adjust, # 在音频处理时再进行范围限制
                fade_in_ms=max(0, min(10000, fade_in_ms)),  # 限制在0-10秒
                fade_out_ms=max(0, min(10000, fade_out_ms)), # 限制在0-10秒
                processing_reason=str(processing_design.get("processing_reason", ""))
            )
            
            return processed_segment
            
        except Exception as e:
            print(f"处理阶段 {phase.phase} 时发生错误: {e}")
            return None

    def design_music_processing(self, phases: List[ProgramPhase], 
                               phase_candidates: Dict[str, List[MusicCandidate]]) -> List[ProcessedMusicSegment]:
        raise NotImplementedError("design_music_processing is not implemented")
        """使用LLM为每个阶段智能选择音乐并设计处理方案（串行版本，作为回退）"""
        processed_segments = []
        
        # 汇总所有阶段简介，帮助模型全局把握与过渡
        phases_overview_lines = []
        for idx, ph in enumerate(phases, 1):
            phases_overview_lines.append(
                f"{idx}. {ph.phase} | duration={ph.duration_seconds}s | purpose={ph.purpose} | criteria={json.dumps(ph.search_criteria, ensure_ascii=False)}"
            )
        phases_overview = "\n".join(phases_overview_lines)
        
        for phase in phases:
            candidates = phase_candidates.get(phase.phase, [])
            if not candidates:
                continue
                
            # 构建LLM提示词，让其选择最佳音乐并设计处理方案
            system_prompt = (
                "You are an expert in GIM therapy and audio processing. Given a therapy phase and music candidates, select the best segment and design appropriate audio processing parameters. Output ONLY a valid JSON that strictly matches this schema (no extra text, no comments, no trailing commas):\n\n"
                "{\n"
                "  \"start_time_seconds\": <float, seconds from beginning, >= 0>,\n"
                "  \"duration_seconds\": <float, seconds, > 0>,\n"
                "  \"speed_ratio\": <float, 1.0=normal, 0.8=20% slower, 1.2=20% faster>,\n"
                "  \"pitch_shift\": <int, semitones, 0=no change, +2=2 semitones higher, -1=1 semitone lower>,\n"
                "  \"volume_adjust\": <float, 1.0=normal, 1.5=50% louder, 0.7=30% quieter>,\n"
                "  \"fade_in_ms\": <int, milliseconds, typically 1000-5000, 0 if not needed>,\n"
                "  \"fade_out_ms\": <int, milliseconds, typically 1000-5000, 0 if not needed>,\n"
                "  \"processing_reason\": <string, concise rationale for choices>\n"
                "}\n\n"
                "Rules:\n"
                "- Return JSON only. Do not include explanations or any additional text outside the JSON.\n"
                "- Choose a musically coherent segment that supports therapeutic continuity and smooth transitions.\n"
                "- Avoid unnecessary edits (use speed_ratio=1.0, pitch_shift=0, volume_adjust=1.0 if already appropriate)."
            )
            
            # 准备候选音乐信息
            candidates_info = []
            for i, candidate in enumerate(candidates):
                temp_metadata = {}
                #只保存filename, tags, genre，mood，movement，theme， audio features中的tempo dynamics_rmse_mean
                temp_metadata["filename"] = candidate.filename
                temp_metadata["tags"] = candidate.metadata.get("tags", [])
                temp_metadata["genre"] = candidate.metadata.get("genre", [])
                temp_metadata["mood"] = candidate.metadata.get("mood", [])
                temp_metadata["movement"] = candidate.metadata.get("movement", [])
                temp_metadata["theme"] = candidate.metadata.get("theme", [])
                temp_metadata["tempo"] = candidate.metadata.get("tempo", [])
                temp_metadata["dynamics_rmse_mean"] = candidate.metadata.get("dynamics_rmse_mean", [])

                info = {
                    "index": i,
                    "title": candidate.title,
                    "duration": candidate.duration_seconds,
                    "metadata": temp_metadata or {}
                }
                candidates_info.append(info)
            ##todo：优化user_prompt，可以放入所有的phase简介，帮助更好的过渡
            user_prompt = (
                "All Phases Overview (for better global planning and transitions):\n" +
                phases_overview +
                "\n\n" +
                f"Current Phase: {phase.phase}\n" +
                f"Required duration: {phase.duration_seconds} seconds\n" +
                f"Music candidate information:\n{json.dumps(candidates_info, ensure_ascii=False, indent=2)}\n\n" +
                "Instructions:\n"
                "- Select the most suitable segment from a candidate to support therapeutic continuity and smooth transitions.\n"
                "- Output ONLY the JSON defined in the system message. No extra text.\n"
                "- If the original segment already fits well, prefer keeping speed_ratio=1.0, pitch_shift=0, volume_adjust=1.0.\n"
            )

            ###prepare the music input
            filepath = candidates[0].filename
            audio_format = self._get_audio_format(filepath)
            audio_b64 = self._encode_base64_content_from_file(filepath)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            
            print("!!music messages: ", system_prompt)
            print("!!music user_prompt: ", user_prompt)
            response_text = self._call_llm(MUSIC_MODEL_NAME, messages)
            processing_design = self._extract_json(response_text)
            
            if not processing_design or not isinstance(processing_design, dict):
                # 兜底：选择第一个候选项，使用默认参数
                selected_candidate = candidates[0]
                processing_design = {
                    "selected_index": 0,
                    "start_time": 0.0,
                    "duration": min(phase.duration_seconds, selected_candidate.duration_seconds),
                    "speed_ratio": 1.0,
                    "pitch_shift": 0.0,
                    "volume_adjust": 1.0,
                    "fade_in_ms": 1000,
                    "fade_out_ms": 1500,
                    "processing_reason": "Default selection due to LLM processing failure"
                }
            
            # 验证并应用选择
            selected_index = processing_design.get("selected_index", 0)
            if 0 <= selected_index < len(candidates):
                selected_candidate = candidates[selected_index]
                
                # 确保参数合理性
                start_time = max(0, float(processing_design.get("start_time", 0)))
                duration = min(
                    phase.duration_seconds,
                    float(processing_design.get("duration", phase.duration_seconds)),
                    selected_candidate.duration_seconds - start_time
                )
                
                # 参数范围验证和默认值设置
                speed_ratio = float(processing_design.get("speed_ratio", 1.0))
                pitch_shift = float(processing_design.get("pitch_shift", 0.0))
                volume_adjust = float(processing_design.get("volume_adjust", 1.0))
                fade_in_ms = int(processing_design.get("fade_in_ms", 1000))
                fade_out_ms = int(processing_design.get("fade_out_ms", 1500))
                
                processed_segment = ProcessedMusicSegment(
                    filename=selected_candidate.filename,
                    title=selected_candidate.title,
                    start_time_seconds=start_time,
                    duration_seconds=duration,
                    speed_ratio=speed_ratio,  # 在音频处理时再进行范围限制
                    pitch_shift=pitch_shift,   # 在音频处理时再进行范围限制
                    volume_adjust=volume_adjust, # 在音频处理时再进行范围限制
                    fade_in_ms=max(0, min(10000, fade_in_ms)),  # 限制在0-10秒
                    fade_out_ms=max(0, min(10000, fade_out_ms)), # 限制在0-10秒
                    processing_reason=str(processing_design.get("processing_reason", ""))
                )
                
                processed_segments.append(processed_segment)
                print(f"阶段 '{phase.phase}' 选择音乐: {selected_candidate.title}")
                print(f"处理方案: {processed_segment.processing_reason}")
        
        return processed_segments

    # ------------------------
    # Step 5: 高级音频合成与编辑
    # ------------------------
    def synthesize_advanced_audio(self, processed_segments: List[ProcessedMusicSegment], 
                                 output_filename: str = "gim_program_mix.mp3",
                                 crossfade_ms: int = 2000) -> Tuple[str, int]:
        """高级音频合成，支持精细化的音频处理"""
        if not which("ffmpeg") and not which("ffmpeg.exe"):
            raise RuntimeError("需要系统已安装 ffmpeg 才能读取和导出 MP3。请先安装 ffmpeg 并确保在 PATH 中。")

        compiled: AudioSegment = None
        total_ms = 0
        processing_log = []

        for idx, seg in enumerate(processed_segments):
            if not seg.filename or not os.path.exists(seg.filename):
                print(f"跳过不存在的文件: {seg.filename}")
                continue

            try:
                # 1. 加载音频文件
                audio = AudioSegment.from_file(seg.filename)
                original_duration = len(audio) / 1000.0
                
                print(f"处理阶段 {idx+1}: {seg.title}")
                print(f"  原始时长: {original_duration:.1f}s")
                
                # 2. 截取指定片段（支持从中间截取）
                start_ms = int(seg.start_time_seconds * 1000)
                duration_ms = int(seg.duration_seconds * 1000)
                end_ms = min(start_ms + duration_ms, len(audio))
                
                if start_ms >= len(audio):
                    print(f"  警告: 开始时间超出音频长度，跳过")
                    continue
                
                clip = audio[start_ms:end_ms]
                print(f"  截取: {seg.start_time_seconds:.1f}s - {end_ms/1000:.1f}s (长度: {len(clip)/1000:.1f}s)")
                
                # 3. 速度调整（使用ffmpeg atempo，支持减速）
                if abs(seg.speed_ratio - 1.0) > 0.01:
                    # 确保速度在合理范围内：0.25-4.0
                    safe_speed = max(0.25, min(4.0, float(seg.speed_ratio)))
                    clip = self._ffmpeg_tempo(clip, safe_speed)
                    print(f"  速度调整: {seg.speed_ratio:.2f}x (应用: {safe_speed:.2f}x)")
                
                # 4. 音调调整（使用ffmpeg asetrate+atempo组合）
                if abs(seg.pitch_shift) > 0.1:
                    # 确保音调调整在合理范围内：-12到+12半音
                    safe_pitch = max(-12.0, min(12.0, float(seg.pitch_shift)))
                    clip = self._ffmpeg_pitch_shift(clip, safe_pitch)
                    direction = "升高" if safe_pitch > 0 else "降低"
                    print(f"  音调调整: {seg.pitch_shift:.1f} 半音 ({direction})")
                
                # 5. 音量调整（dB = 20*log10(gain)）
                if abs(seg.volume_adjust - 1.0) > 0.01:
                    # 确保音量在合理范围内：0.1-3.0
                    safe_volume = max(0.1, min(3.0, float(seg.volume_adjust)))
                    try:
                        gain_db = 20.0 * math.log10(safe_volume)
                    except Exception:
                        gain_db = 0.0
                    clip = clip.apply_gain(gain_db)
                    direction = "增大" if safe_volume > 1.0 else "减小"
                    print(f"  音量调整: {seg.volume_adjust:.2f}x ({direction}, {gain_db:+.1f}dB)")
                
                # 6. 淡入淡出（限制不超过片段长度的30%）
                max_fade = int(len(clip) * 0.3)
                fade_in_ms = min(max(0, int(seg.fade_in_ms)), max_fade)
                fade_out_ms = min(max(0, int(seg.fade_out_ms)), max_fade)
                if fade_in_ms or fade_out_ms:
                    clip = clip.fade_in(fade_in_ms).fade_out(fade_out_ms)
                print(f"  淡入淡出: {fade_in_ms}ms / {fade_out_ms}ms")
                
                # 7. 拼接到主音轨（交叉渐变不超过两端片段长度）
                if compiled is None:
                    compiled = clip
                else:
                    effective_crossfade = min(crossfade_ms, len(compiled) - 1, len(clip) - 1) if len(compiled) > 1 and len(clip) > 1 else 0
                    if effective_crossfade > 0:
                        compiled = compiled.append(clip, crossfade=effective_crossfade)
                        print(f"  交叉渐变: {effective_crossfade}ms")
                    else:
                        compiled = compiled + clip
                        print(f"  交叉渐变: 0ms (片段过短，改为直接拼接)")
                
                total_ms += len(clip)
                
                # 记录处理日志
                processing_log.append({
                    "title": seg.title,
                    "start_time": seg.start_time_seconds,
                    "duration": seg.duration_seconds,
                    "speed_ratio": seg.speed_ratio,
                    "pitch_shift": seg.pitch_shift,
                    "volume_adjust": seg.volume_adjust,
                    "reason": seg.processing_reason
                })
                
            except Exception as e:
                print(f"处理音频片段 {seg.title} 时出错: {e}")
                continue

        if compiled is None:
            print("[WARN] No audio segments available")
            return None
        # 8. 导出最终音频
        out_path = os.path.join(self.output_dir, output_filename)
        compiled.export(out_path, format="mp3")
        
        print(f"\n音频合成完成:")
        print(f"  输出文件: {out_path}")
        print(f"  总时长: {total_ms/1000:.1f}s")
        print(f"  处理了 {len(processing_log)} 个音频片段")
        
        return out_path, total_ms
    
    # 保留原始的简单合成方法作为后备
    def synthesize_audio(self, segments: List[MusicSegment], output_filename: str = "gim_program_mix.mp3",
                          fade_in_ms: int = 1000, fade_out_ms: int = 1500, crossfade_ms: int = 2000,
                          speed_ratio: float = None) -> Tuple[str, int]:
        """简单的音频合成方法（保持向后兼容）"""
        if not which("ffmpeg") and not which("ffmpeg.exe"):
            raise RuntimeError("需要系统已安装 ffmpeg 才能读取和导出 MP3。请先安装 ffmpeg 并确保在 PATH 中。")

        compiled: AudioSegment = None
        total_ms = 0

        for idx, seg in enumerate(segments):
            if not seg.filename or not os.path.exists(seg.filename):
                continue

            audio = AudioSegment.from_file(seg.filename)

            # 可选速度调整（谨慎使用）
            if speed_ratio and speed_ratio > 0 and abs(speed_ratio - 1.0) > 1e-3:
                # 使用内置的speedup（会改变音高；如果需要不变调，可集成librosa或sox）
                audio = speedup(audio, playback_speed=speed_ratio, chunk_size=50, crossfade=0)

            # 截取所需时长
            need_ms = max(0, seg.duration_seconds * 1000)
            clip = audio[:need_ms] if len(audio) >= need_ms else audio

            # 渐变
            clip = clip.fade_in(fade_in_ms).fade_out(fade_out_ms)

            # 拼接
            if compiled is None:
                compiled = clip
            else:
                compiled = compiled.append(clip, crossfade=crossfade_ms)
            total_ms += len(clip)

        if compiled is None:
            print("[WARN] No audio segments available")
            return None

        out_path = os.path.join(self.output_dir, output_filename)
        compiled.export(out_path, format="mp3")
        return out_path, total_ms

    # ------------------------
    # 高层封装：从对话到智能合成
    # ------------------------
    def build_and_render_program(self, chat_history: List[Dict[str, str]], 
                                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """完整的GIM Program构建和渲染流程
        
        Args:
            chat_history: 对话历史
            progress_callback: 进度回调函数，接收参数:
                - stage: 当前阶段 ("analysis", "design", "music_search", "processing", "synthesis")
                - status: 状态信息
                - progress: 进度值 (0-100)
                - data: 相关数据
        """
        def update_progress(stage: str, status: str, progress: float = 0, data: Any = None):
            if progress_callback:
                progress_callback(stage, status, progress, data)
            print(f"[{stage}] {status} - {progress}%")
        
        print("=== 开始GIM Program构建流程 ===")
        update_progress("analysis", "开始分析对话内容...", 0)
        
        # Step 1: 对话理解与治疗意图分析
        analysis = self.analyze_dialogue(chat_history)
        update_progress("analysis", "分析完成", 100, {
            "emotion": analysis.get('current_emotion', '未知'),
            "goal": analysis.get('therapeutic_goal', '未设定'),
            "themes": analysis.get('key_themes', [])
        })
        
        # Step 2: 设计Program框架
        update_progress("design", "开始设计治疗Program...", 0)
        phases = self.design_program(analysis)
        update_progress("design", "设计完成", 100, {
            "phase_count": len(phases),
            "phases": [{
                "name": p.phase,
                "duration": p.duration_seconds,
                "purpose": p.purpose
            } for p in phases]
        })
        
        # Step 3: 检索音乐候选项
        update_progress("music_search", "开始检索音乐...", 0)
        phase_candidates = self.retrieve_music_candidates(phases, candidates_per_phase=CANDIDATES_PER_PHASE)
        update_progress("music_search", "检索完成", 100, {
            "candidates": {
                phase: [{"title": c.title, "duration": c.duration_seconds} for c in candidates]
                for phase, candidates in phase_candidates.items()
            }
        })
        
        # Step 4: 智能选择和处理设计（使用并行版本）
        update_progress("processing", "开始处理音乐...", 0)
        
        # 创建一个进度追踪的回调
        processed_count = 0
        total_phases = len([p for p in phases if phase_candidates.get(p.phase)])
        
        def processing_progress(phase_name: str, status: str, is_complete: bool = False):
            nonlocal processed_count
            if is_complete:
                processed_count += 1
            progress = (processed_count / total_phases) * 100 if total_phases > 0 else 100
            update_progress("processing", f"处理阶段: {phase_name} - {status}", progress)
        
        processed_segments = self.design_music_processing_parallel(
            phases, 
            phase_candidates,
            progress_callback=processing_progress
        )
        
        update_progress("processing", "音乐处理完成", 100, {
            "segments": [{
                "title": seg.title,
                "duration": seg.duration_seconds,
                "processing": {
                    "speed": seg.speed_ratio,
                    "pitch": seg.pitch_shift,
                    "volume": seg.volume_adjust,
                    "fade_in": seg.fade_in_ms,
                    "fade_out": seg.fade_out_ms
                }
            } for seg in processed_segments]
        })
        
        # Step 5: 高级音频合成
        update_progress("synthesis", "开始音频合成...", 0)
        output_path, total_ms = self.synthesize_advanced_audio(processed_segments)
        
        # 构建详细的结果报告
        program_json = {
            "analysis": analysis,
            "program": [
                {
                    "phase": p.phase,
                    "duration_seconds": p.duration_seconds,
                    "purpose": p.purpose,
                    "search_criteria": p.search_criteria,
                }
                for p in phases
            ],
            "candidates": {
                phase: [
                    {
                        "title": c.title,
                        "filename": c.filename,
                        "duration_seconds": c.duration_seconds,
                    }
                    for c in candidates
                ]
                for phase, candidates in phase_candidates.items()
            },
            "processed_segments": [
                {
                    "title": s.title,
                    "filename": s.filename,
                    "start_time_seconds": s.start_time_seconds,
                    "duration_seconds": s.duration_seconds,
                    "speed_ratio": s.speed_ratio,
                    "pitch_shift": s.pitch_shift,
                    "volume_adjust": s.volume_adjust,
                    "processing_reason": s.processing_reason,
                }
                for s in processed_segments
            ],
            "output": {
                "file": output_path,
                "total_seconds": total_ms // 1000,
            },
        }
        
        print("=== GIM Program构建完成 ===")
        return program_json
    
    # 保留简化版本作为后备
    def build_and_render_program_simple(self, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """简化的Program构建流程（向后兼容）"""
        analysis = self.analyze_dialogue(chat_history)
        phases = self.design_program(analysis)
        
        # 使用简化的音乐检索
        segments = []
        for phase in phases:
            criteria = phase.search_criteria or {}
            tracks = self.music_db.retrieve_music_for_therapy(criteria, num_tracks=1) or []
            if tracks:
                chosen = tracks[0]
                filename = chosen.get("filename", "")
                title = chosen.get("title", filename)
                
                if filename and not os.path.isabs(filename):
                    candidate = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "../toy_dataset/mp3", os.path.basename(filename))
                    )
                    if os.path.exists(candidate):
                        filename = candidate
                        
                segments.append(MusicSegment(filename=filename, duration_seconds=phase.duration_seconds, title=title))
        
        output_path, total_ms = self.synthesize_audio(segments)

        program_json = {
            "analysis": analysis,
            "program": [
                {
                    "phase": p.phase,
                    "duration_seconds": p.duration_seconds,
                    "purpose": p.purpose,
                    "search_criteria": p.search_criteria,
                }
                for p in phases
            ],
            "segments": [
                {
                    "title": s.title,
                    "filename": s.filename,
                    "duration_seconds": s.duration_seconds,
                }
                for s in segments
            ],
            "output": {
                "file": output_path,
                "total_seconds": total_ms // 1000,
            },
        }
        return program_json

    # ------------------------
    # 工具：LLM 调用与 JSON 提取
    # ------------------------

    def _call_llm(self, model_name, messages: List[Dict[str, str]], timeout: int = 60) -> str:
        """调用LLM模型，带有超时控制
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            timeout: 超时时间（秒），默认60秒
        """
        payload = json.dumps({
            "model": model_name,
            "max_tokens": MAX_TOKENS_RESPONSE,
            "temperature": 0.2,
            "messages": messages,
        })
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Connection': 'close'
        }
        
        # 重试策略：最多3次，指数退避
        last_err = None
        for attempt in range(3):
            try:
                # 使用超时设置
                conn, api_path = make_llm_connection(timeout=timeout)
                start_time = time.time()
                
                try:
                    conn.request("POST", api_path, payload, headers)
                    response = conn.getresponse()
                    raw = response.read()
                    
                    # 检查是否超时
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"LLM调用超时（{timeout}秒）")
                    
                    try:
                        text = raw.decode("utf-8")
                    except Exception:
                        text = raw.decode("utf-8", errors="ignore")
                    
                    try:
                        response_data = json.loads(text)
                        if isinstance(response_data, dict) and 'choices' in response_data:
                            return response_data['choices'][0]['message']['content']
                        elif isinstance(response_data, dict) and 'content' in response_data:
                            return response_data['content'][0]['text']
                        return ""
                    except json.decoder.JSONDecodeError:
                        # 可能是网关返回HTML或空响应，重试
                        last_err = f"Invalid JSON: {text[:200]}"
                        
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                        
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, 
                    http.client.ResponseNotReady, socket.timeout, TimeoutError) as e:
                last_err = str(e)
                print(f"LLM调用出错（尝试 {attempt + 1}/3）: {e}")
            except Exception as e:
                last_err = str(e)
                print(f"LLM调用未知错误（尝试 {attempt + 1}/3）: {e}")
            
            # 退避等待
            wait_time = min(1.5 * (attempt + 1), 5)  # 最多等待5秒
            print(f"等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)
        
        print(f"LLM调用失败: {last_err}")
        return ""

    def _extract_json(self, text: str) -> Any:
        if not text:
            return None
        # 提取代码块中的 JSON
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)
        # 尝试解析
        try:
            return json.loads(text)
        except Exception:
            # 尝试提取第一个 { .. }
            brace_match = re.search(r"\{[\s\S]*\}", text)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except Exception:
                    return None
            # 或者 [ .. ]
            bracket_match = re.search(r"\[[\s\S]*\]", text)
            if bracket_match:
                try:
                    return json.loads(bracket_match.group(0))
                except Exception:
                    return None
            return None 
