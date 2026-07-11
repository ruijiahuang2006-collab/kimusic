## 3 System Design

Figure 1 shows the GIMFlow architecture. The system is organized as a browser-facing client and a GIMFlow server that coordinates six modules: a *Session Manager*, a *Therapy Agent* implementing a four-phase LLM dialogue, an *Emotion/VA Planner*, a *Music Module* with two counterbalanced conditions, a *Questionnaire Module*, and a *Logger* that exports full session-level JSON. User messages, LLM outputs, planning artifacts, music metadata, and questionnaire responses are written to a common log to support offline analysis and blinded evaluation.

![Figure 1: GIMFlow system architecture. Two feature tabs (top) highlight the phase-scripted Therapy Agent (four dialogue phases coordinated by an LLM state classifier) and the Emotion/VA Planner coupled to the Music Module (illustrative planner values are drawn from the paper case study in §5.5). The Music Module operates in one of two within-subject counterbalanced conditions: Kimusic (LLM blueprint → deterministic MIDI rendering) or Baseline (emotion metadata → library retrieval). Three supporting modules — Session Manager, Questionnaire Module (SAM, PANAS, SUS, TX), and Logger — coordinate on a shared session state and export one JSON per session.](fig1_architecture.png)

### 3.1 Therapy Workflow

GIMFlow scripts a four-phase dialogue mirroring the structural arc of a GIM session \cite{grocke_guided_2019}. **Prelude** elicits the user's current concern and emotional state through open-ended questions. **Induction** provides a relaxation-oriented text prompt that scaffolds entry into imagery. **Music & Imaging** invites the user to trigger playback of a selected or generated music piece and to describe the imagery that arises. **Postlude** integrates the imagery experience into a brief reflective summary. Each phase is realized by a phase-specific LLM prompt and returns explicit phase and speaker labels in the log, allowing downstream analysis to compare content and affect across phases. Figure 2 shows representative Session 1 UI screenshots.

![Figure 2: Representative Session 1 UI screenshots. ① Prelude elicits the user's emotional state and focus theme; ② Music & Imaging supports user-triggered playback of the generated or retrieved piece; the right panel displays the LLM-generated music plan with four program phases, including each phase's duration and intended purpose; ③ Post-session assessment collects post-SAM (post-PANAS is on a subsequent screen); ④ Session transition displays the participant ID and enforces a 5-minute washout countdown before Session 2.](fig2_ui.png)

### 3.2 Emotion Analysis

GIMFlow represents affective state in the valence-arousal (VA) circumplex \cite{russell_circumplex_1980, posner_circumplex_2005}. Between phases, the system queries the LLM to estimate the user's current VA coordinates and a target VA state consistent with the stated concern (for example, relaxation, calming). The LLM returns continuous valence and arousal values rather than a categorical label, drawing on the emotion-recognition-in-conversation literature and its LLM-based instantiations \cite{poria_emotion_2019, zhang_dialoguellm_2025}. The LLM additionally emits a short natural-language *focus theme* string (for example, *"fatigue and desire for rest"*) that supplies human-interpretable affective context to the downstream music module.

### 3.3 Waypoint Planning

Given the estimated current VA and target VA, the VA Planner constructs a waypoint sequence connecting the two endpoints. The current implementation places four waypoints along a straight line via linear interpolation. The planner is deterministic and does not personalize the trajectory to user history or adapt during a session. The explicit waypoint representation is intended as an enabler for future policy-based extensions rather than a claim of sophisticated planning in the current release. Figure 3 illustrates a representative trajectory.

![Figure 3: Illustrative VA trajectory used by the VA Planner. The trajectory begins at a tense/anxious current state and moves through two intermediate waypoints toward a relaxed/content target. Coordinates are constructed for illustration and are not derived from participant data.](fig3_va_trajectory.png)

### 3.4 Music Module: Generation and Retrieval

The Music Module operates in one of two counterbalanced conditions sharing a unified schema. Downstream logs are equivalent in structure but differ only in the source of the audio.

**Kimusic (blueprint-conditioned symbolic rendering).** The Kimusic condition prompts an LLM to produce a symbolic music *blueprint* specifying the piece's overall parameters (tempo, key, harmonic language, melodic motion, dynamic curve, texture, and instrument roles) and a section-level structural plan. A procedural renderer built with `midiutil`, `fluidsynth`, and a General MIDI soundfont produces a per-session audio file from the blueprint (overall tempo, key, chord progression by key signature, instrument roles, and a section-level dynamics profile). "Generation" here means session-specific blueprint-conditioned symbolic rendering, not neural text-to-audio synthesis in the sense of \cite{agostinelli_musiclm_2023, copet_simple_2024, melechovsky_mustango_2024}. Two versions of the Kimusic condition are relevant to §5:

- *Initial deployment (§5.1).* The blueprint's affective content aligned only with the endpoint (target) waypoint; the three intermediate waypoints influenced the surrounding dialogue prompts but were not mapped onto the audio.
- *Current refined deployment (§5.3 onward).* The blueprint's `structure` field contains four sections, each linked to one waypoint (`section_i.waypoint_index = i`, `section_i.waypoint_va = waypoint_i.va`), so that the planned affective trajectory unfolds across intra-piece structural sections.

All other GIMFlow components (Session Manager, Therapy Agent, VA Planner, Questionnaire Module, Logger) are identical across the two deployments. Sessions are associated with session-specific rendered audio paths, and file-level uniqueness across generated audio was checked during system validation.

**Baseline (retrieval).** The Baseline condition selects a piece from an indexed library of instrumental tracks with associated mood, genre, tempo, and instrumentation metadata. The LLM emits two criteria sets: a *positive* set of retrieval keywords and a *negative* set of avoid keywords, matched against track metadata. Each retrieved baseline track is logged with a rule-derived `emotion_label` computed from the retrieval criteria (primarily the mood keywords) and with the retrieval keywords themselves. This design follows emotion-aware music recommendation approaches \cite{tran_emotion-aware_2023, wang_novel_2021}. Because the current baseline library does not include track-level VA annotations, no VA distance is computed for baseline sessions; conditions are compared at the level of session-level affect and self-report, not track-level VA alignment.

**Scope of the within-subject comparison.** The two conditions share every component except the Music Module. Findings favoring one condition should therefore be interpreted as evidence about *blueprint-conditioned symbolic rendering versus metadata retrieval as music-delivery pathways within an otherwise identical workflow*, not as evidence about the value of the surrounding system.

### 3.5 Dialogue Agent

The Therapy Agent is a prompt-driven wrapper around GPT-5.5. For each user turn, the agent determines the current phase from the session state, issues an internal VA-estimation request to the emotion analyzer, and emits a phase-appropriate reply generated from a phase-specific prompt template. The agent supports both Chinese and English inputs. GIMFlow is best characterized as a phase-scripted workflow agent whose action space is fixed by the workflow phases and their prompt templates.

### 3.6 Session Logging

Every session emits a single JSON record with six field groups: identifiers, pre- and post-session questionnaires, dialogue, emotion planning and music, usability and experience, and completion tracking. Appendix E describes the schema in detail. The JSON is finalized at completion. Both a `post_session_complete` snapshot (used when the participant closes the questionnaire step) and a `complete` snapshot (used when the participant additionally completes the post-experiment questionnaires) are supported. For analysis, the most complete snapshot per session ID is retained.

---