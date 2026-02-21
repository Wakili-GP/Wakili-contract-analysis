---
title: Contract Analyzer API
emoji: 📜
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
app_file: main.py
pinned: false
license: apache-2.0
---

# Contract Analyzer API

API لتحليل العقود القانونية بالذكاء الاصطناعي (RAG + LLM)  
- يدعم رفع PDF/DOCX  
- يعطي ملخص + مخاطر + التزامات + جدول مقارنة  
- دردشة متابعة مع السياق من العقد  

**Endpoints الرئيسية:**
- POST /analyze → رفع العقد + تحليل شامل  
- POST /chat → أسئلة متابعة  

**كيفية الاستخدام:**
- استخدمي الرابط /docs للواجهة التجريبية  
- أضيفي GROQ_API_KEY في الـ Variables عشان يشتغل الـ LLM

المشروع مبني بـ FastAPI + LangChain + Chroma + Groq