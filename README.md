# Local Developer Telemetry & Ergonomics Engine

A local, privacy-first Python pipeline that correlates system usage with biometric focus metrics. It uses computer vision to track screen attention and posture in real-time, aggregates the data locally using Pandas, and leverages Gemini (via LangChain) to generate actionable productivity insights.

## Features
* **Zero-Leakage Biometrics:** Gaze and posture are calculated locally on your machine using MediaPipe. No video feeds are ever sent over the network.
* **Smart Aggregation:** High-frequency per-second tracking data is bucketed into hourly summaries via Pandas, preventing LLM context-window exhaustion and reducing API costs.
* **Ergonomic Alerts:** Triggers native Windows audio alerts if poor posture or screen distraction exceeds an active threshold.
* **LLM Analytics Engine:** Feeds deterministic telemetry matrices into Gemini 2.5 Flash for high-level productivity profiling.

## Prerequisites
* Python 3.9+
* Windows OS (for `winsound` audio alerts)
* A valid Google Gemini API Key

## Installation

1. **Clone the repository and set up a virtual environment:**
   ```cmd
   git clone [https://github.com/yourusername/local-telemetry-engine.git](https://github.com/yourusername/local-telemetry-engine.git)
   cd local-telemetry-engine
   python -m venv openclaw-env
   .\openclaw-env\Scripts\activate
