# 🤖 DecoBot — Rule-Based AI Chatbot
### DecodeLabs | AI Internship 2026 | Project 1

---

## 📌 Project Overview

DecoBot is a rule-based AI chatbot built as the **foundation milestone** of the DecodeLabs AI Internship (Batch 2026). It demonstrates core AI engineering principles — control flow, deterministic decision-making, and the IPO (Input → Process → Output) model — without relying on any machine learning or external libraries.

---

## 🎯 Objective

> Build a simple rule-based chatbot that responds to predefined user inputs using if-else logic, running in a continuous loop.

---

## ✅ Features

| Feature | Description |
|---|---|
| 🔁 Infinite Loop | Continuous `while True` cycle keeps the bot alive |
| 🧹 Input Sanitization | `.lower().strip()` normalizes all user input |
| 📚 Knowledge Base | Dictionary with 10+ intents for O(1) lookup |
| 🔍 Keyword Scan | Partial matching for more natural conversations |
| 💬 Fallback Response | Graceful default for unknown inputs |
| 🚪 Exit Strategy | Clean `break` on `quit / exit / bye / goodbye` |
| 🛡️ Empty Input Guard | Handles blank input without crashing |

---

## 🏗️ Architecture

The bot follows the **IPO Model** taught in the DecodeLabs training kit:

```
USER INPUT
    │
    ▼
┌─────────────────────┐
│  PHASE 1: SANITIZE  │  → lower() + strip()
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  PHASE 2: MATCH     │  → Dictionary lookup → Keyword scan → Fallback
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  PHASE 3: RESPOND   │  → Print reply → Loop back
└─────────────────────┘
```

### Why Dictionary over If-Elif Ladder?

| Approach | Complexity | Scalability |
|---|---|---|
| If-Elif Ladder | O(n) — gets slower with more rules | ❌ High technical debt |
| Dictionary `.get()` | O(1) — instant lookup always | ✅ Clean and scalable |

---

## 📁 Project Structure

```
project-1/
│
├── chatbot.py      # Main chatbot script
└── README.md       # Project documentation
```

---

## ⚙️ Requirements

- Python 3.x
- No external libraries required

---

## 🚀 How to Run

1. Clone or download the project folder.
2. Open a terminal in the project directory.
3. Run the following command:

```bash
python chatbot.py
```

4. Start chatting! Type `help` to see available topics.
5. Type `quit` or `exit` to end the session.

---

## 💬 Sample Conversation

```
=======================================================
   Welcome to DecoBot | DecodeLabs AI Internship 2026
   Type 'help' for options  |  Type 'quit' to exit
=======================================================

You   : hello
DecoBot : Hey there! I'm DecoBot 🤖. How can I help you today?

You   : what is ai
DecoBot : AI (Artificial Intelligence) is the simulation of human
          intelligence by machines — from simple rule-based logic
          to deep learning models.

You   : who are you
DecoBot : I'm DecoBot — a rule-based AI chatbot built at DecodeLabs.
          Pure logic, zero hallucinations. 😎

You   : quit
DecoBot : Goodbye! Keep coding and stay curious. 🚀
=======================================================
```

---

## 🧠 Concepts Demonstrated

- **Control Flow** — while loops, if-else, break, continue
- **Data Structures** — Python dictionaries as a knowledge base
- **String Methods** — `.lower()`, `.strip()`, `in` operator
- **IPO Model** — Input sanitization → Intent matching → Output generation
- **Algorithmic Efficiency** — O(1) dictionary lookup vs O(n) if-elif
- **White-Box AI** — Every decision is fully traceable and explainable

---

## 🔮 Possible Extensions

- Add more intents to expand the bot's vocabulary
- Implement nested conditions for context-aware responses
- Add response randomization (multiple replies per intent)
- Give the bot a unique personality or domain focus
- Use this as the rule-based guardrail layer over an LLM (Project 2+)

---

## 👨‍💻 Author

- **Name:** Dhyan Shah
- **Batch:** DecodeLabs AI Internship 2026
- **Organization:** DecodeLabs, Greater Lucknow, India
---

> *"An LLM without rules is a hallucination engine. Today, we build the skeleton that holds the intelligence."*