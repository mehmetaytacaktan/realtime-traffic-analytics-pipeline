# 🚘🚘 Realtime Traffic Analytics Pipeline 🚘🚘

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://senin-app-linkin.streamlit.app](https://realtime-traffic-analytics-pipeline-iu3kqeb6wqmcb2riz6jq42.streamlit.app/))
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=flat)](#)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

An end-to-end Computer Vision pipeline designed for real-time vehicle detection, multi-object tracking (MOT), and dynamic line-crossing counting. Powered by **YOLOv8**, **ByteTrack**, **Supervision**, and **Streamlit**, with integrated **FFmpeg H.264 video codec optimization** for web rendering.

 **Live Interactive Demo:** [Try the application on Streamlit Cloud](https://realtime-traffic-analytics-pipeline-iu3kqeb6wqmcb2riz6jq42.streamlit.app/)
---

##  Key Features

- **Vehicle Detection & Classification:** High-precision inference using YOLOv8 tuned for transport classes (Car, Motorcycle, Bus, Truck).
- **Multi-Object Tracking (MOT):** Consistent object ID assignment across frames utilizing **ByteTrack**.
- **Line-Crossing Analytics:** Automated inbound/outbound vehicle counting powered by `supervision.LineZone`.
- **Web-Compatible Video Transcoding:** Automated post-processing via **FFmpeg** (`libx264` / `yUV420p` color space) to resolve browser playback inconsistencies.
- **Interactive UI & Cloud Deployment:** Built with Streamlit for dynamic file uploads and sample testing, fully containerized via **Docker**.

---
