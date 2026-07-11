# Demo Video Recording Guide

**Watch the video**: https://youtu.be/lompQnNoJlM (2:16, YouTube unlisted)

This document describes the recording procedure for the GIMFlow demonstration video accompanying the EMNLP 2026 System Demonstrations submission. The video shows a complete session walkthrough covering all four dialogue phases (Prelude, Induction, Music & Imaging, Postlude), music generation, and pre/post assessments.

## Video Specifications

- **Total length**: approximately 2:15 (within the CFP 2:30 limit)
- **Format**: mp4, YouTube unlisted upload
- **Narration**: live voice-over by the first author (Rita Huang)
- **Screencast target**: the deployed GIMFlow web interface
- **Language**: English (system dialogue and narration)
- **Music playback**: at least 15 seconds of audible generated music included

## Content Scope

The video covers:
- One complete session with the Kimusic (LLM-guided blueprint-conditioned) condition
- All four dialogue phases: Prelude, Induction, Music & Imaging, Postlude
- Pre-session SAM and PANAS assessments (briefly shown)
- Post-session SAM and PANAS assessments (briefly shown)
- Program Panel display of blueprint content
- Audible music playback
- Session 2 protocol and post-experiment measures mentioned verbally, not shown

The video does not cover:
- Participant background survey (not part of the technical contribution)
- Consent form (Study 2 procedure is described in the Ethics Statement of the paper)
- Full multi-turn Music & Imaging exchange (a shortened representative exchange is shown; a fuller session transcript is available in `example_session_transcript.md`)

## Section Breakdown

### Section 1 — Opening [0:00-0:15]

Introduction to the system:

> "Hi, I'm Rita. This is GIMFlow, a non-clinical research demonstration that combines an LLM-guided dialogue workflow with symbolic music rendering. Let me walk you through one complete session."

### Section 2 — Prelude and Induction

**2a. Pre-session assessments.** SAM and PANAS forms are shown briefly:

> "The session opens with SAM and PANAS assessments."

**2b. Prelude dialogue.** The Prelude exchange between the system and the participant is displayed, elicting current emotional state and a focus theme.

> "The Prelude phase elicits current state and a focus theme. I'll respond as a participant. The LLM extracts a focus theme and estimates valence-arousal coordinates. That drives the Music Module."

**2c. Induction.** The opening lines of the Induction phase are shown:

> "The Induction phase then guides the participant through a brief relaxation. Music generation runs in parallel."

### Section 3 — Music generation (cut-away)

> "Music generation takes about thirty seconds. I'll cut ahead. Behind the scenes, the LLM emits a JSON blueprint specifying key, harmonic progression, and a four-section structure aligned to the target trajectory. A procedural renderer then converts that blueprint into MIDI and audio."

At "I'll cut ahead," the video cuts directly to the frame showing music generation completion.

### Section 4 — Music & Imaging

**4a. Program Panel.** The Program Panel displays the blueprint content:

> "The Program Panel on the right shows what the blueprint contains: a therapeutic goal string, the detected emotional state, and the four-section program with durations. Total piece length is ninety seconds."

**4b. Music playback and imagery.** The music player is activated. The participant's imagery message is typed:

> "Let me play the piece. As the music unfolds, the participant is invited to describe any imagery. I'll show that."

Music plays audibly for approximately 15 seconds, during which the participant enters a first imagery message. The system responds by extending and reflecting the imagery.

### Section 5 — Postlude and post-session assessments

> "When the music ends, the Postlude phase then integrates the experience into a brief reflective summary, brings awareness back to the room, and closes the session. The system auto-detects Postlude completion and closes the session, then presents post-session SAM and PANAS assessments."

### Section 6 — Session 2 protocol and resources

> "Session 2 follows the same four-phase protocol with the alternate condition, following a five-minute washout period. After both sessions, participants complete SUS and TX questionnaires. GIMFlow was evaluated in two within-subject studies. Paper, code, and demo audio are linked in the video description. Thanks for watching."

## Visual Highlights

In the produced video, key dialogue and interface elements are visually highlighted (red circles or boxes) to draw reviewer attention to the following:

1. **Prelude participant input** ("I've been feeling drained and heavy this week. My mind won't settle.") — the input that triggers focus theme extraction and VA estimation
2. **Program Panel** — the therapeutic goal string, detected emotional state, and four-section program with durations, representing the LLM-produced music blueprint
3. **Music & Imaging participant imagery** ("A path bathed in moonlight, I am all alone.") — the participant's imagery report during music playback
4. **Postlude integration summary** — the system's synthesis of the Prelude focus theme with the imagery arising during Music & Imaging

## Note on Deployed vs Released Interface Wording

The demonstrated deployed interface uses some therapist-oriented labels inherited from the practical origins of GIM as a therapy method, including "Therapeutic Goal" and the four section names shown in the Program Panel ("Grounding and Arrival", "Breathing and Relaxation", "Reflective Wellness Imagery", "Integration and Uplift"). The released prompts distributed with the code repository use non-clinical wording aligned with the paper's framing. This distinction is disclosed in §8 (Limitations) and Appendix A of the paper.

## Note on Participant Dialogue

The participant lines shown in the video correspond to a demonstration scenario. A fuller session transcript covering the extended Music & Imaging exchange and Postlude reflection is provided separately in `example_session_transcript.md` for reviewers or reproducers interested in the full facilitator dialogue style.

## Ethical Considerations

The recorded session is a demonstration performed by the first author acting as a participant, not a real research session. No real participant data is shown at any point. The demonstration follows the same protocol structure used in the studies to give reviewers a faithful sense of the participant experience, without exposing any real participant material.

The consent form and background survey used in Study 2 are not shown in the video. The Study 2 consent procedure (formal written consent, exclusion criteria, and crisis-support resources) is described in the paper's Ethics Statement.
