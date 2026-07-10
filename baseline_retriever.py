from __future__ import annotations
import os
import random


def _dedupe(values):
    result = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _cluster_from_keywords(criteria):
    keywords = [keyword.lower() for keyword in criteria.get("mood_keywords", [])]
    clusters = {
        "restorative": [
            "restorative", "peaceful", "warm", "calming", "calm",
            "relaxing", "relax", "soft", "gentle", "dreamy",
            "meditation", "soothing", "sleep", "quiet", "piano"
        ],
        "energetic": [
            "energetic", "uplifting", "hopeful", "energizing",
            "optimistic", "bright", "happy", "joyful", "fun",
            "inspiring", "positive", "epic"
        ],
        "tense": [
            "tense", "anxious", "restless", "suspense",
            "dramatic", "dark", "intense", "frantic",
            "scary", "tension"
        ],
        "melancholic": [
            "melancholic", "reflective", "fragile",
            "sad", "sentimental", "emotional", "nostalgia",
            "lonely", "romantic", "heartbreaking"
        ],
        "neutral": [
            "balanced", "gentle", "neutral",
            "calm", "soft", "relax", "relaxing",
            "peaceful", "dreamy", "morning",
            "instrumental", "classical", "piano",
            "background", "beautiful", "hopeful"
        ],
    }

    for cluster, cluster_keywords in clusters.items():
        if any(keyword in keywords for keyword in cluster_keywords):
            return cluster
    return "neutral"


def retrieve_baseline_track(
    music_db,
    criteria,
    track_index=0,
    used_track_ids=None,
    music_root="../toy_dataset/mp3",
):
    criteria = criteria or {}
    used_track_ids = set(used_track_ids or [])

    candidates = music_db.retrieve_music_for_therapy(criteria, num_tracks=12)

    if not candidates:
        emotion_label = _cluster_from_keywords(criteria)
        print(f"[WARN] No baseline tracks for {emotion_label}, using relaxed fallback")

        fallback_criteria = {
            "mood_keywords": [
                "calm", "soft", "relaxing", "peaceful", "dreamy",
                "hopeful", "bright", "classical", "piano", "background"
            ],
            "genre_keywords": ["solo piano", "modern classical", "classical"],
            "avoid_keywords": [],
            "tempo_preference": "medium",
            "dynamics_preference": "moderate",
        }

        candidates = music_db.retrieve_music_for_therapy(fallback_criteria, num_tracks=12)

    if not candidates:
        print("[WARN] Relaxed fallback also failed, using general fallback")
        fallback_criteria = {
            "mood_keywords": [],
            "genre_keywords": [],
            "avoid_keywords": [],
        }
        candidates = music_db.retrieve_music_for_therapy(fallback_criteria, num_tracks=12)

    if not candidates:
        raise ValueError("Baseline fallback failed: no tracks returned from music database")

    available = []
    for track in candidates:
        track_id = track.get("track_id") or track.get("id") or track.get("filename") or track.get("title")
        if track_id not in used_track_ids:
            available.append(track)

    if not available:
        available = candidates

    track = random.choice(available).copy()
    filename = track.get("filename")

    # 强制使用服务器真实 mp3 目录，不再使用 JSON 里的旧 file_path
    relative_path = os.path.join(os.path.dirname(__file__), "../toy_dataset/mp3", filename) if filename else track.get("file_path")
    full_path = os.path.abspath(relative_path) if relative_path else None

    print("DEBUG full_path =", full_path)
    track_id = track.get("track_id") or track.get("id") or filename or f"baseline_track_{track_index}"

    retrieval_keywords = _dedupe(
        criteria.get("mood_keywords", [])
        + criteria.get("genre_keywords", [])
    )
    avoid_keywords = _dedupe(criteria.get("avoid_keywords", []))

    return {
        **track,
        "track_id": track_id,
        "id": track_id,
        "source": "database",
        "music_source": "database",
        "file_path": relative_path,
        "full_path": full_path,
        "filename": filename or os.path.basename(relative_path or ""),
        "title": track.get("title") or filename or track_id,
        "track_title": track.get("title") or filename or track_id,
        "emotion_label": _cluster_from_keywords(criteria),
        "retrieval_keywords": retrieval_keywords,
        "avoid_keywords": avoid_keywords,
    }
