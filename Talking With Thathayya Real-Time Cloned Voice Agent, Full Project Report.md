# Talking With Thathayya: Real-Time Cloned Voice Agent

*Full project report: from first recording to a live, natural conversation. Prepared October 16, 2026.*

## 1. Honest Goals and Expectations

The goal is a live, phone-call-style conversation with an AI that speaks in your grandfather's cloned voice, in his Telugu dialect, with his personality and memories.

What is realistic with today's technology:

- **Voice:** a good clone can sound very close to him, especially with clean, long recordings.
- **Speed:** a response time of 0.8 to 1.5 seconds is achievable. This feels like a natural phone call.
- **Personality:** this depends mostly on the quality of the stories, phrases, and memories you collect, not on the AI model.
- **Limit:** it will not be *exactly* him. The AI generates new sentences he never said, so it is an approximation built from his voice and your family's memories. Plan for this emotionally. Many families find it comforting; some find it painful. Decide as a family what role it should have.
- **Free:** the whole project can be built for free or nearly free using free tiers and open-source tools. Free tiers have limits, and they change often, so verify each one before you rely on it.

## 2. Consent, Ethics, and Safety (Do This First)

1. If your grandfather is alive, ask his permission and record it (a short video of him saying yes).
2. If he has passed away, discuss it with your close family before building.
3. Never use the voice to deceive anyone, make real phone calls pretending to be him, or ask for money or information.
4. Keep the voice model, recordings, and API keys private. Do not share the voice ID publicly.
5. Add a gentle limit: for example, tell the agent in its prompt never to give medical, legal, or financial advice as if it were him.
6. Never paste real API keys in chats or code. Use a `.env` file and add it to `.gitignore`. Rotate any key you have exposed.

## 3. System Architecture

```
Microphone -> VAD (detects speech) -> Streaming STT (Telugu) -> text
text + memories -> LLM (his personality) -> reply text, streamed
reply text -> Streaming TTS (cloned voice) -> audio, streamed
audio -> Speaker
(barge-in: if you speak, stop audio and listen again)
```

Every stage must **stream**. Each stage starts work before the previous one finishes. This is what makes it feel like a call instead of a walkie-talkie.

### Latency budget (target under about 1.2 seconds)

| Stage | Target |
| --- | --- |
| End-of-speech detection | 200 to 400 ms |
| Speech-to-text final result | 100 to 300 ms |
| LLM first sentence | 300 to 500 ms |
| TTS first audio | 100 to 200 ms |
| Network and playback | 100 ms |

The single `/tts/bytes` request you tested waits for the full audio, so it is for testing only. Use Cartesia's streaming (WebSocket) endpoint for the live agent.

## 4. Technology Stack (Free-First)

| Layer | Best option | Free alternative / note |
| --- | --- | --- |
| Orchestration | **Pipecat** (open source) | LiveKit Agents (open source) is also excellent |
| Voice clone + TTS | **Cartesia** (you already have the clone) | Free tier has limited credits. Open-source backup: AI4Bharat Indic voice models that support Telugu (check current repos and licenses) |
| Speech-to-text | **Deepgram** or **Google** streaming STT (test Telugu quality) | **faster-whisper** (open source, run locally; use a larger model for Telugu, a GPU helps) |
| LLM brain | **Claude** (best for personality and nuance) | Google Gemini or Groq free tiers for testing |
| Voice activity detection | **Silero VAD** (free, built into Pipecat) | none needed |
| Memory | JSON/Markdown files first, then a small vector DB (**Chroma**, free) |  |
| App / transport | Browser with WebRTC (Pipecat or LiveKit) | Phone number later with Twilio (paid) |
| Hosting | Your own laptop for development | Free tiers on Render, Fly.io, or Hugging Face Spaces for demos |

**Important:** I cannot verify current free-tier limits, model names (such as `sonic-3.6`) or API versions. Check each provider's pricing and docs page before you start.

## 5. Phase-by-Phase Plan

### Phase 0: Preparation (Days 1 to 3)

- Install Python 3.11 or newer, Git, and VS Code.
- Create accounts: Cartesia, Anthropic (Claude), Deepgram, and GitHub.
- Create the project folder:

```
thathayya-agent/
  .env
  .gitignore
  requirements.txt
  agent.py
  persona/
    system_prompt.md
    memories.md
    phrases.md
  data/
    recordings/
  tests/
```

- Put `.env` in `.gitignore` before your first commit.

### Phase 1: Collect and Prepare the Voice (Days 1 to 5)

Voice quality decides everything else.

- Gather **10 to 30 minutes** of his clearest speech if you can. Cartesia's cloning can work from much less, but more clean audio gives better results. Check Cartesia's current guidance.
- Prefer: one speaker only (him), no music, no echo, no TV or family noise, natural speaking, and a variety of emotions (greeting, laughing, telling a story, scolding lovingly).
- Remove other voices and noise with free tools: **Audacity** (noise reduction), or **Ultimate Vocal Remover** for separating background music.
- Save as WAV, mono, 44.1 kHz or 48 kHz.
- Make a short **test script** of 10 to 15 sentences in his style, including his favorite phrases, and check the clone against them.
- If the clone sounds wrong, record better samples and re-clone. Do not try to fix a bad clone with prompts.

### Phase 2: Pronunciation and Dialect Dictionary (Days 3 to 7)

This answers your original question about accurate pronunciation and slang.

1. Transcribe his recordings. Use faster-whisper for a draft, then **correct every line by hand** with a family member who knows how he speaks. The transcript must match what he actually said, including dialect spellings.
2. Build a word list: the words he says in a distinctive way (for example, Telangana-style endings and vocabulary, or his personal habits).
3. Write each entry like this:

| Word (Telugu script) | How he says it (your spelling) | Notes |
| --- | --- | --- |
| (word) | (how it sounds) | Where he uses it, any variation |

4. Use this list for two things: (a) the **system prompt**, so the LLM writes in his dialect, and (b) Cartesia's pronunciation features if your plan supports them, to fix words the TTS mispronounces.
5. Native-speaker review is the only reliable accuracy check. An AI cannot hear his accent from a text description.

### Phase 3: Build the Persona (Days 5 to 10)

This is what makes the *words* feel like him.

**system\_prompt.md should include:**

- Who he is: name, age, where he grew up, profession, family members and relationships.
- How he addresses you (pet names), how he greets and says goodbye.
- Speech style: short sentences, favorite sayings, jokes, how he reacts when happy, worried, or teasing.
- Dialect rules: words to use and avoid, from your Phase 2 list.
- Topics he loves and topics he avoids.
- Boundaries: keep replies to 1 to 3 short sentences, ask questions back, do not invent big life facts, and if unsure say so in his way (for example, "I do not remember, tell me again").
- Safety: never give medical, legal, or money instructions.

**memories.md should include:** real stories, dates, places, names, favorite food, songs, festivals, and advice he often gave. Write them in his own words if possible. Add 20 to 50 good items to start.

**phrases.md should include:** 50 to 100 real sentences he said, with context. Put 10 to 20 of the best in the prompt as few-shot examples. This is the strongest technique for style.

### Phase 4: Text Chat Prototype (Days 8 to 12)

Build the brain first, without audio.

- Write a small script: type a message, send the persona prompt plus relevant memories to Claude, print the reply.
- Have family members chat with it and score it: Does it sound like him? What would he never say?
- Update the prompt and memories every round. Expect 5 to 10 rounds.

### Phase 5: Voice Output (Days 10 to 14)

- Test the `/tts/bytes` request with your own cloned voice ID, saving to `output.wav`.
- Then switch to Cartesia **streaming** over WebSocket. Send text sentence by sentence as the LLM streams, and play audio chunks as they arrive.
- Tune the speed and emotion settings if your model supports them. Test 20 sentences and note which sound unnatural.

### Phase 6: Voice Input (Days 12 to 18)

- Add microphone capture, Silero VAD, and streaming STT with Telugu (and English if your family mixes languages).
- **Test with real family speech.** STT often struggles with regional accents, mixed Telugu and English, and background noise. Compare at least three providers on the same 20 recorded sentences and pick the lowest error rate.
- Use a headset during development to prevent the agent from hearing its own voice (echo).

### Phase 7: Real-Time Pipeline with Pipecat (Days 15 to 25)

Use Pipecat's pipeline: transport (browser WebRTC) -> VAD -> STT -> LLM -> TTS -> transport.

Key features to configure:

- **Barge-in:** stop speaking within about 200 ms when you start talking.
- **End-of-turn detection:** wait long enough that it does not cut you off mid-sentence, but not so long that it feels slow. Elders and thoughtful speakers pause; tune for that.
- **Filler sounds:** a short "hmm..." or "ఆ..." if the LLM takes longer than about 1 second. Generate these in his voice ahead of time and play them from files.
- **Streaming by sentence:** break LLM output at sentence boundaries and send each to TTS immediately.
- **Short replies:** natural conversation has short turns.
- **Memory:** keep the last 10 to 20 turns in context, and retrieve the most relevant memories per turn.

### Phase 8: Naturalness Polish (Days 25 to 35)

- Add small personality touches: he asks about your day, remembers what you said earlier, and uses his pet names.
- Vary reply length and avoid repeating the same phrases.
- Add short pauses and laughs in his voice where his real recordings have them.
- Run "blind" tests: have relatives listen to samples of real versus generated speech and note what feels off. Fix the top three issues each round.
- Test with poor internet, loud rooms, and different devices.

### Phase 9: Delivery (Days 35 to 45)

- **Browser app (best first step):** a simple web page with a Call button, hosted privately with a password.
- **Phone app feel:** make the page a Progressive Web App so it opens like an app on a phone.
- **Real phone number (optional, paid):** use Twilio Media Streams to connect calls to the pipeline. Only family should be able to call.
- Add logging with consent, so you can find problems, but keep transcripts private and delete them on request.

## 6. Quality Metrics and Test Checklist

| Metric | Target | How to test |
| --- | --- | --- |
| Response latency | under 1.2 s average | Log timestamps from end of speech to first audio |
| Interruption response | under 300 ms | Speak while it talks |
| STT accuracy (Telugu) | as high as possible; compare providers | 20 recorded sentences, count errors |
| Voice similarity | family rating 8/10 or higher | Blind listening test |
| Personality realism | family rating 8/10 or higher | Chat sessions and score sheet |
| Stability | 30-minute call without failure | Long test |

## 7. Cost Overview

- **Development:** can be near zero using free tiers and local open-source tools.
- **Regular use:** the real costs are TTS characters, LLM tokens, and STT minutes. Estimate them from each provider's current pricing page. A few short family calls per week should be modest, but long daily calls can add up.
- **Money-saving tips:** keep replies short, cache common phrases as audio files, use a cheaper LLM for simple turns, and run STT locally if you have a GPU.

## 8. Risks and Fixes

| Risk | Fix |
| --- | --- |
| Clone sounds robotic or off | Re-clone with cleaner, longer, more varied audio |
| STT misunderstands Telugu | Compare providers, add custom vocabulary, use a headset, test a Whisper model fine-tuned for Telugu |
| Responses feel generic | Add more real phrases, stories, and dialect rules; use few-shot examples |
| Latency too high | Stream everything, use a nearby server region, shorten prompts, use a faster LLM |
| AI says something he never would | Add "never say" rules, review logs, update the prompt |
| Emotional impact | Agree on healthy limits as a family; it is a memory tool, not a replacement |
| Service changes or shutdown | Keep local backups of recordings, transcripts, and persona files |

## 9. Weekly Timeline Summary

| Week | Goal |
| --- | --- |
| 1 | Ethics, accounts, project setup, collect and clean recordings |
| 2 | Clone voice, build dialect dictionary, write persona v1 |
| 3 | Text chat prototype, family feedback, voice output tested |
| 4 | Voice input, compare STT providers |
| 5 | Full real-time Pipecat pipeline with barge-in |
| 6 | Naturalness polish and blind tests |
| 7 | Browser app, privacy and security, family launch |

## 10. Your Next Three Actions

1. Ask permission (or talk with family) and gather his best audio recordings.
2. Re-check the Cartesia clone against 10 sentences, and write down what sounds wrong.
3. Write persona v1 (system prompt, 20 memories, 30 phrases). Then ask me to build the text chat prototype, then the Pipecat voice pipeline.

## 11. Final Note

No AI model or tool can promise "exactly the same as the real person." What you can build is a warm, fast, convincing voice conversation that carries his voice, his words, and your family's stories. The more real material you collect, the closer it gets. Start small, test with family, and improve each week.
