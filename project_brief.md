# Project Brief: ScholarPage — Professor Paper Interactive Site Creator

## 1. Overview & Core Hypothesis
**Hypothesis**: Academic professors and researchers want their published papers to reach a wider audience, gain more citations, and drive real-world practice. Traditional academic PDFs are dense, static, and difficult for industry practitioners, journalists, and students to digest.

**Solution**: **ScholarPage** is a dedicated, verified platform for professors that automatically transforms dense academic papers into beautiful, no-code, interactive web pages.

---

## 2. Core Value Proposition & Key Features
1. **Verified Academic Studio**:
   - Verified professor portal (Institutional ORCID / Edu Login verification).
2. **Instant Paper Transformation Engine**:
   - Upload PDF, paste DOI, or paste ArXiv link.
   - Gemini AI automatically parses abstract, core thesis, key takeaways, methodology, figures, and citation metadata.
3. **No-Code Customization Controls**:
   - **Information Density Slider**:
     - *Executive Summary / Industry Brief* (high condensation, plain English, zero jargon).
     - *Balanced* (key figures + core logic + practical impact).
     - *Full Academic Rigor* (complete methodology + mathematical derivations).
   - **Visualization Mode Toggle**:
     - *Beautified Interactive Charts* (Chart.js / SVG animated graphs with hover tooltips).
     - *Static Academic Figures* (original vector figures).
   - **Target Audience Tuning**:
     - Industry Practitioners, Policy Makers, Students, or Peer Researchers.
4. **Knowledge Translation & Engagement**:
   - **"Ask the Paper" AI Chatbot Widget**: Visitors can ask questions directly against the paper context.
   - **1-Click BibTeX / APA Citation Copy**.
   - **Download Practical PDF Brief**.
   - **Embeddable Widget**: Professors can embed their interactive paper page into personal faculty sites or LinkedIn.

---

## 3. User Experience & Architecture
* **Frontend**: Responsive, modern HTML5/CSS3/JavaScript web interface using Substrate/Scholar aesthetic (clean typography, sleek glass cards, dark/light modes, live preview).
* **Backend Agent**: ADK Python Agent powered by Gemini 2.5 Flash for PDF extraction, paper summarization, chart data generation, and interactive Q&A.
* **Hosting**: Cloud Run deployment with public HTTPS URL.
