# GIMFlow

## Naming

**GIMFlow** refers to the overall interactive guided-imagery music system.

**Kimusic** refers specifically to the blueprint-based music generation module and to the experimental Kimusic condition. The repository name and selected implementation identifiers retain `kimusic` for compatibility with the original module structure.

**An LLM-Guided GIM-Inspired Music Agent**

*System Demonstration submission to EMNLP 2026*

GIMFlow is a non-clinical research demonstration that combines an LLM-guided dialogue workflow with symbolic music rendering. The system enacts a phase-scripted four-phase dialogue (Prelude, Induction, Music & Imaging, Postlude), estimates the user's current valence–arousal state, plans a target-directed four-waypoint trajectory, and delivers music via either blueprint-conditioned symbolic rendering (Kimusic condition) or metadata-based retrieval (Baseline condition).

---

## Public Release Scope

This repository is a sanitized public release for the EMNLP Demo submission. Private participant data, raw session logs, API keys, generated outputs, and copyrighted baseline audio assets are intentionally excluded.

The public code preserves the GIMFlow system implementation and the Kimusic music-generation module. The full Baseline condition requires a private local music database and is available only in the secured server deployment.

## Demo Materials

- **Video walkthrough** (2:16, YouTube unlisted): https://youtu.be/lompQnNoJlM
- **Full demonstration transcript**: [`docs/example_session_transcript.md`](docs/example_session_transcript.md)
- **Video recording guide**: [`docs/demo_video_recording_guide.md`](docs/demo_video_recording_guide.md)
- **Demo audio**: [`demo_audio/`](demo_audio/)

## Paper

- **Paper (OpenReview)**: [TBD after submission]
- **Live demo**: [TBD]

---

## Repository Structure

```
kimusic/
├── app.py
├── prompts.py                              # Released prompts with non-clinical wording
├── kimusic_renderer.py                     # Blueprint -> MIDI/audio renderer
├── kimusic_client.py                       # Blueprint generation client
├── baseline_retriever.py                   # Baseline retrieval logic
├── music_db.py                             # Baseline metadata support
├── ai_evaluation/
│   ├── build_pairwise_inputs_from_session_results.py
│   ├── run_ai_evaluation.py
│   └── analyze_ai_evaluations_swapped.py
├── docs/
│   ├── example_session_transcript.md
│   ├── demo_video_recording_guide.md
│   ├── case_study_note.md
│   ├── v5_appendix_A_demographics.md
│   ├── v5_appendix_C_questionnaire_items.md
│   ├── v5_appendix_D_full_prompts.md
│   ├── v5_appendix_G_split_analysis.md
│   └── v5_section3_full_system_design.md
├── data/
│   └── aggregate_analysis/
├── figures/
├── demo_audio/
├── soundfonts/
│   └── FluidR3_GM_LICENSE.txt
├── LICENSE
└── README.md
```

---

## Getting Started

Create a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your API endpoint and model identifiers, then run:

```bash
python app.py
```

---

## Environment Variables

See [`.env.example`](.env.example). The released code is intended to work with an OpenAI-compatible endpoint by setting `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and model identifiers such as `DIALOGUE_MODEL` and `AI_EVAL_MODEL`.

---

## Notes for Reviewers

### Deployed vs Released Interface Wording

The deployed interface used during the studies (and shown in the demonstration video) includes some therapist-oriented labels inherited from the practical origins of Guided Imagery and Music (GIM) as a therapy method. These labels include:

- "Therapeutic Goal" (top of the Program Panel)
- Section names such as "Grounding and Arrival", "Relaxation and Breath Awareness", "Gentle Introspection", and "Integration and Positive Closure"

The **released prompts** distributed in this repository under `prompts.py` use **non-clinical wording** aligned with the paper's framing. This distinction is disclosed in §8 (Limitations) and Appendix A of the paper. Future deployments will use the non-clinical wording consistently.

### Models Used in Paper Experiments

The paper experiments used GPT-5.5 for the dialogue, valence–arousal estimation, and blueprint generation components, and Claude Sonnet 4.6 for the LLM judge in the AI evaluation. The released code is provider-agnostic: set `DIALOGUE_MODEL` and `AI_EVAL_MODEL` in `.env` to the model identifiers offered by your chosen provider.

### Study 1 vs Study 2 Procedural Differences

Study 1 (formative pilot, N=23) used a written recruitment notice describing purpose and voluntary participation, but did not include a formal written consent form, explicit pre-participation exclusion criteria, or crisis-support resources. Study 2 (preliminary evaluation of refined system, N=10) introduced a formal written informed consent form, four pre-participation exclusion criteria, and crisis-support hotline resources. Full details are described in §9 (Ethics Statement) of the paper.

### Ethical Framing

GIMFlow is a **research demonstration and not a substitute for professional mental-health care**. The recorded session shown in the video is a demonstration performed by the first author acting as a participant, not a real research session, and no real participant data is shown in the video.

---

## Data Availability

Individual session data, raw questionnaire responses, and raw LLM-judge outputs are not publicly released to protect participant privacy.

Aggregate results are provided in `data/aggregate_analysis/`, including means, standard deviations, paired effect sizes, and Wilcoxon signed-rank test results for Study 1 and Study 2.

For the case study discussed in §5.5 of the paper, extended details are provided in the paper appendix. The repository includes `docs/case_study_note.md` to clarify the scope of released case-study materials and the privacy rationale for not releasing individual session-level logs.

---

## Audio and SoundFont Attribution

GIMFlow demo audio is provided in `demo_audio/` when available. The FluidR3_GM SoundFont license text is provided in `soundfonts/FluidR3_GM_LICENSE.txt`. Demo audio rendered with this SoundFont is distributed subject to the terms in that file.

The baseline audio library is not redistributed because track-level redistribution rights cannot be uniformly guaranteed under the original sources' licensing terms.

---

## License

Released under the [MIT License](LICENSE).

---

## Citation

If accepted, citation information will be added here after publication.

For now, please refer to the OpenReview submission page (link above) for the current paper version.

---

## Contact

Ruijia (Rita) Huang — ruijia.huang@mail.utoronto.ca


## Key Repository Artifacts

The following files provide the main reviewer-facing artifacts referenced by the paper:

- Full prompt documentation: docs/v5_appendix_D_full_prompts.md
- Kimusic blueprint construction interface: docs/v5_appendix_D2_kimusic_blueprint_interface.md
- LLM judge prompt: docs/v5_appendix_F_llm_judge_prompt.md
- Synthetic example session JSON: data/example_session/example_session_synthetic.json
- Aggregate Study 1/2 results: data/aggregate_analysis/
- Demo audio sample: demo_audio/demo_session_audio.mp3
- High-resolution figures: figures/

## Language Note

This public release is intended for English-language demonstration, review, and reproducibility. Some legacy Chinese UI strings from the deployed study interface are not maintained in the public release code; the paper, README, documentation, demo transcript, prompts, and reviewer-facing artifacts are provided in English.

